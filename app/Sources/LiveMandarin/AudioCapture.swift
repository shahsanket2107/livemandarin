import CoreMedia
import Foundation
import ScreenCaptureKit

/// Captures system audio (every app except ourselves) with ScreenCaptureKit and emits
/// 100 ms frames of 16 kHz mono Int16 PCM — the caption server's input format.
final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    enum CaptureError: LocalizedError {
        case noDisplay
        var errorDescription: String? { "No display found to attach the audio capture to." }
    }

    var onFrame: ((Data, Float) -> Void)?   // 100 ms Int16 PCM frame, peak level 0…1
    var onError: ((Error) -> Void)?
    private(set) var framesDelivered = 0

    private var stream: SCStream?
    private let queue = DispatchQueue(label: "captions.audio")
    private var pending: [Int16] = []
    private static let frameSamples = 1600

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { throw CaptureError.noDisplay }

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        // Audio-only: shrink the (unused) video side to nothing.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false

        let stream = SCStream(filter: SCContentFilter(display: display, excludingWindows: []),
                              configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stop() async {
        guard let stream else { return }
        self.stream = nil
        try? await stream.stopCapture()
        queue.async { self.pending.removeAll() }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        try? sampleBuffer.withAudioBufferList { bufferList, _ in
            guard let buffer = bufferList.first, let data = buffer.mData else { return }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            let samples = data.bindMemory(to: Float.self, capacity: count)
            pending.reserveCapacity(pending.count + count)
            for i in 0..<count {
                let clamped = max(-1, min(1, samples[i]))
                pending.append(Int16(clamped * 32767))
            }
        }
        while pending.count >= Self.frameSamples {
            let samples = pending[0..<Self.frameSamples]
            let frame = samples.withUnsafeBufferPointer { Data(buffer: $0) }
            let peak = Float(samples.reduce(0) { max($0, abs(Int32($1))) }) / 32767
            pending.removeFirst(Self.frameSamples)
            framesDelivered += 1
            onFrame?(frame, peak)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        self.stream = nil
        onError?(error)
    }
}
