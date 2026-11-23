// Web client: microphone or file -> WebSocket (real time), or file -> server-sent events (as fast as possible).
(() => {
  const $ = (id) => document.getElementById(id);
  const I18N = {
    zh: {
      subtitle: '英中双向同声传译', direction: '方向', en2zh: '英文 → 中文', zh2en: '中文 → 英文',
      micStart: '开始麦克风', stop: '结束并收尾', realtime: '按实时速度播放', upload: '翻译文件', copy: '复制',
      clear: '清空', idle: '就绪', connecting: '连接中…', live: '同传中', finishing: '收尾中…', done: '完成',
      busy: '服务繁忙', error: '出错', noFile: '请先选择音频文件', noMic: '无法使用麦克风',
      hint: '远程访问时浏览器只允许在 HTTPS 或 localhost 下使用麦克风；可用 SSH 端口转发后打开 http://127.0.0.1:端口/。',
      audio: '音频', step: '每步', skipped: '跳过块',
    },
    en: {
      subtitle: 'English ↔ Chinese simultaneous translation', direction: 'Direction', en2zh: 'English → Chinese',
      zh2en: 'Chinese → English', micStart: 'Start microphone', stop: 'Finish', realtime: 'Play in real time',
      upload: 'Translate file', copy: 'Copy', clear: 'Clear', idle: 'Ready', connecting: 'Connecting…',
      live: 'Translating', finishing: 'Finishing…', done: 'Done', busy: 'Server busy', error: 'Error',
      noFile: 'Choose an audio file first', noMic: 'Microphone unavailable',
      hint: 'Browsers allow the microphone only on HTTPS or localhost; for a remote server, forward the port over SSH and open http://127.0.0.1:PORT/.',
      audio: 'audio', step: 'step', skipped: 'skipped blocks',
    },
  };
  let lang = 'zh';
  try { lang = localStorage.getItem('simulst-lang') || (navigator.language.startsWith('zh') ? 'zh' : 'en'); } catch (e) { /* storage blocked */ }
  const t = (k) => (I18N[lang] && I18N[lang][k]) || k;
  function applyLang() {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    $('lang').textContent = lang === 'zh' ? 'EN' : '中文';
  }
  $('lang').onclick = () => {
    lang = lang === 'zh' ? 'en' : 'zh';
    try { localStorage.setItem('simulst-lang', lang); } catch (e) { /* ignore */ }
    applyLang();
  };
  applyLang();

  const WORKLET = `
    class Tap extends AudioWorkletProcessor {
      process(inputs) {
        const chans = inputs[0];
        if (chans && chans.length) {
          const n = chans[0].length, out = new Float32Array(n);
          for (const c of chans) for (let i = 0; i < n; i++) out[i] += c[i] / chans.length;
          this.port.postMessage(out, [out.buffer]);
        }
        return true;
      }
    }
    registerProcessor('tap', Tap);`;

  let ws = null, ctx = null, tap = null, src = null, media = null, pending = [], pendingLen = 0, busy = false;

  function setStatus(key, on) {
    $('status').textContent = t(key);
    $('status').classList.toggle('on', !!on);
  }
  function setBusy(b) {
    busy = b;
    $('mic').disabled = b;
    $('upload').disabled = b;
    $('direction').disabled = b;
    $('stop').disabled = !b;
    $('mic').classList.toggle('live', b);
  }
  function show(ev) {
    if (ev.event === 'translation') {
      const box = $('text');
      box.textContent = ev.text;
      box.scrollTop = box.scrollHeight;
      $('stats').textContent = `${t('audio')} ${ev.audio_s.toFixed(1)} s · ${t('step')} ${Math.round(ev.step_ms)} ms`;
    } else if (ev.event === 'done') {
      $('text').textContent = ev.text;
      const parts = [`${t('audio')} ${ev.audio_s.toFixed(1)} s`];
      if (ev.step_ms_p50 != null) parts.push(`${t('step')} p50 ${Math.round(ev.step_ms_p50)} ms / p90 ${Math.round(ev.step_ms_p90)} ms`);
      if (ev.skipped_blocks) parts.push(`${t('skipped')} ${ev.skipped_blocks}`);
      $('stats').textContent = parts.join(' · ');
      setStatus('done');
      teardown();
    } else if (ev.event === 'error') {
      $('stats').textContent = ev.message;
      setStatus(/busy/.test(ev.message) ? 'busy' : 'error');
      teardown();
    }
  }
  function wsUrl() {
    const u = new URL('translate/ws', location.href);
    u.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return u.toString();
  }
  function flush(force) {
    if (!ws || ws.readyState !== WebSocket.OPEN || !pendingLen) return;
    if (!force && pendingLen < ctx.sampleRate / 10) return;   // send ~100 ms at a time
    const buf = new Float32Array(pendingLen);
    let o = 0;
    for (const p of pending) { buf.set(p, o); o += p.length; }
    pending = []; pendingLen = 0;
    ws.send(buf.buffer);
  }
  async function startAudio(makeSource) {
    ctx = new AudioContext();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await ctx.audioWorklet.addModule(url);
    tap = new AudioWorkletNode(ctx, 'tap');
    const mute = ctx.createGain();
    mute.gain.value = 0;
    tap.connect(mute).connect(ctx.destination);
    tap.port.onmessage = (e) => { pending.push(e.data); pendingLen += e.data.length; flush(false); };
    src = await makeSource(ctx);
    src.connect(tap);
  }
  function openSession(makeSource) {
    return new Promise((resolve, reject) => {
      setBusy(true);
      setStatus('connecting');
      $('text').textContent = '';
      $('stats').textContent = '';
      ws = new WebSocket(wsUrl());
      ws.binaryType = 'arraybuffer';
      ws.onopen = async () => {
        try {
          ctx = null;
          // the sample rate is only known after the audio graph exists
          await startAudio(makeSource);
          ws.send(JSON.stringify({ direction: $('direction').value, sample_rate: ctx.sampleRate, format: 'f32le', channels: 1, realtime: true }));
          resolve();
        } catch (err) { reject(err); }
      };
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data);
        if (ev.event === 'ready') setStatus('live', true); else show(ev);
      };
      ws.onerror = () => { setStatus('error'); teardown(); };
      ws.onclose = () => { if (busy) teardown(); };
    });
  }
  function stopInput() {
    if (media) { media.getTracks().forEach((x) => x.stop()); media = null; }
    if (src) { try { src.disconnect(); if (src.stop) src.stop(); } catch (e) { /* already stopped */ } src = null; }
    flush(true);
    if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify({ eof: true })); setStatus('finishing', true); }
  }
  function teardown() {
    if (media) { media.getTracks().forEach((x) => x.stop()); media = null; }
    if (ctx) { ctx.close().catch(() => {}); ctx = null; }
    if (ws) { const w = ws; ws = null; try { w.close(); } catch (e) { /* ignore */ } }
    pending = []; pendingLen = 0;
    setBusy(false);
  }

  $('mic').onclick = async () => {
    try {
      await openSession(async (c) => {
        media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
        return c.createMediaStreamSource(media);
      });
    } catch (err) {
      $('stats').textContent = `${t('noMic')}: ${err.message || err}`;
      setStatus('error');
      teardown();
    }
  };
  $('stop').onclick = () => stopInput();

  $('upload').onclick = async () => {
    const f = $('file').files[0];
    if (!f) { $('stats').textContent = t('noFile'); return; }
    if ($('realtime').checked) {
      try {
        const data = await f.arrayBuffer();
        await openSession(async (c) => {
          const buf = await c.decodeAudioData(data);
          const s = c.createBufferSource();
          s.buffer = buf;
          s.connect(c.destination);          // play it, and stream what is played
          s.onended = () => stopInput();
          s.start();
          return s;
        });
      } catch (err) {
        $('stats').textContent = String(err.message || err);
        setStatus('error');
        teardown();
      }
      return;
    }
    // as fast as possible: upload and read server-sent events
    setBusy(true);
    $('stop').disabled = true;
    setStatus('live', true);
    $('text').textContent = '';
    $('stats').textContent = '';
    const fd = new FormData();
    fd.append('file', f);
    fd.append('direction', $('direction').value);
    fd.append('realtime', 'false');
    try {
      const resp = await fetch('translate/stream', { method: 'POST', body: fd });
      if (!resp.ok) { show(await resp.json()); return; }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let rest = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        rest += dec.decode(value, { stream: true });
        let i;
        while ((i = rest.indexOf('\n\n')) >= 0) {
          const chunk = rest.slice(0, i);
          rest = rest.slice(i + 2);
          if (chunk.startsWith('data: ')) {
            const ev = JSON.parse(chunk.slice(6));
            if (ev.event !== 'ready') show(ev);
          }
        }
      }
    } catch (err) {
      $('stats').textContent = String(err.message || err);
      setStatus('error');
    } finally {
      setBusy(false);
    }
  };

  $('copy').onclick = () => { navigator.clipboard && navigator.clipboard.writeText($('text').textContent); };
  $('clear').onclick = () => { $('text').textContent = ''; $('stats').textContent = ''; };
})();
