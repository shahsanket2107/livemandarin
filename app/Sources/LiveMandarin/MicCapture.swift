import AVFoundation
import Foundation

/// Optional microphone input (your own voice), converted to 16 kHz mono Int16 and mixed into
/// the system-audio frames by the coordinator. Uses whatever input device macOS has selected
/// (AirPods, built-in mic, …).
final class MicCapture {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private var buffer: [Int16] = []
    private static let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!

    var isRunning: Bool { engine.isRunning }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { throw NSError(domain: "MicCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "No microphone available."]) }
        converter = AVAudioConverter(from: format, to: Self.target)
        input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] pcm, _ in
            self?.convert(pcm)
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.withLock { buffer.removeAll() }
    }

    /// Take up to `count` samples (zero-padded) to mix with a system-audio frame.
    func take(_ count: Int) -> [Int16] {
        lock.withLock {
            let n = min(count, buffer.count)
            var out = Array(buffer[0..<n])
            buffer.removeFirst(n)
            if buffer.count > 16000 { buffer.removeFirst(buffer.count - 16000) }  // never lag more than 1 s
            out.append(contentsOf: repeatElement(0, count: count - n))
            return out
        }
    }

    private func convert(_ pcm: AVAudioPCMBuffer) {
        guard let converter else { return }
        let capacity = AVAudioFrameCount(Double(pcm.frameLength) * 16000 / pcm.format.sampleRate) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: Self.target, frameCapacity: capacity) else { return }
        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return pcm
        }
        guard error == nil, let data = out.int16ChannelData?[0] else { return }
        let samples = Array(UnsafeBufferPointer(start: data, count: Int(out.frameLength)))
        lock.withLock { buffer.append(contentsOf: samples) }
    }
}
