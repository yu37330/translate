// Offline worklet test: 16 kHz and 48 kHz input, no microphone required.
"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const html = fs.readFileSync(__dirname + "/templates/index.html", "utf8");
const match = html.match(/const workletCode = \`([\s\S]*?)\`;/);
assert.ok(match, "AudioWorklet inline code is present");
const source = match[1].replace(/\$\{SAMPLE_RATE\}/g, "16000");

function testRate(rate) {
  let Processor;
  const packets = [];
  class FakeAudioWorkletProcessor {
    constructor() {
      this.port = {postMessage: packet => packets.push(packet)};
    }
  }
  vm.runInNewContext(source, {
    sampleRate: rate,
    AudioWorkletProcessor: FakeAudioWorkletProcessor,
    registerProcessor: (name, ctor) => {
      assert.equal(name, "pcm-collector");
      Processor = ctor;
    },
  });
  assert.ok(Processor);
  const processor = new Processor();
  const input = new Float32Array(128).fill(0.25);
  for (let i = 0; i < 375; i++) {
    assert.equal(processor.process([[input]]), true);
  }
  const count = packets.reduce((sum, p) => sum + p.pcm.length, 0);
  const target = 375 * 128 * 16000 / rate;
  assert.ok(Math.abs(count - target) <= 1, `rate=${rate} samples=${count} expected=${target}`);
  assert.ok(packets.every(p => [...p.pcm].every(x => x > 8100 && x < 8300)));
}

testRate(16000);
testRate(48000);
testRate(44100);
console.log("AudioWorklet resampling tests passed");
