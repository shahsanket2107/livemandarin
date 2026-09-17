// Renders the app icon (rounded gradient tile with a caption bubble) to app/AppIcon.icns.
// Run: swift scripts/make-icon.swift
import AppKit

let root = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().deletingLastPathComponent()
let iconset = root.appendingPathComponent("build/AppIcon.iconset")
try? FileManager.default.removeItem(at: iconset)
try! FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

func render(size: Int) -> NSImage {
    let s = CGFloat(size)
    let image = NSImage(size: NSSize(width: s, height: s))
    image.lockFocus()
    let inset = s * 0.05  // macOS icons leave a margin around the tile
    let tile = NSBezierPath(roundedRect: NSRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset),
                            xRadius: s * 0.2, yRadius: s * 0.2)
    NSGradient(colors: [NSColor(red: 0.13, green: 0.47, blue: 0.98, alpha: 1),
                        NSColor(red: 0.36, green: 0.22, blue: 0.86, alpha: 1)])!
        .draw(in: tile, angle: -60)
    // Glyph: a speech bubble with two caption bars.
    let bubble = NSBezierPath(roundedRect: NSRect(x: s * 0.2, y: s * 0.32, width: s * 0.6, height: s * 0.42),
                              xRadius: s * 0.1, yRadius: s * 0.1)
    bubble.move(to: NSPoint(x: s * 0.36, y: s * 0.33))
    bubble.line(to: NSPoint(x: s * 0.3, y: s * 0.22))
    bubble.line(to: NSPoint(x: s * 0.47, y: s * 0.33))
    bubble.close()
    NSColor.white.withAlphaComponent(0.96).setFill()
    bubble.fill()
    NSColor(red: 0.2, green: 0.4, blue: 0.9, alpha: 1).setFill()
    for (i, w) in [0.38, 0.26].enumerated() {
        NSBezierPath(roundedRect: NSRect(x: s * 0.29, y: s * (0.57 - Double(i) * 0.12), width: s * w, height: s * 0.065),
                     xRadius: s * 0.03, yRadius: s * 0.03).fill()
    }
    image.unlockFocus()
    return image
}

for (name, size) in [("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("128x128", 128),
                     ("128x128@2x", 256), ("256x256", 256), ("256x256@2x", 512), ("512x512", 512), ("512x512@2x", 1024)] {
    let image = render(size: size)
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    image.draw(in: NSRect(x: 0, y: 0, width: size, height: size))
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: iconset.appendingPathComponent("icon_\(name).png"))
}

let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c", "icns", iconset.path, "-o", root.appendingPathComponent("app/AppIcon.icns").path]
try! task.run()
task.waitUntilExit()
try? FileManager.default.removeItem(at: iconset)
print(task.terminationStatus == 0 ? "wrote app/AppIcon.icns" : "iconutil failed")
