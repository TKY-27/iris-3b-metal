import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Image decoding off the main thread, with downsampled results cached by size.
enum ImageFiles {
    private final class Cache: @unchecked Sendable {  // NSCache is thread-safe
        let images = NSCache<NSString, CGImage>()
        init() { images.totalCostLimit = 384 << 20 }
    }

    private static let cache = Cache()

    /// The image at `url`, at most `maxPixels` on its long side and upright.
    static func thumbnail(of url: URL, maxPixels: Int) async -> CGImage? {
        let key = "\(url.path)#\(maxPixels)" as NSString
        if let cached = cache.images.object(forKey: key) { return cached }
        let image = await Task.detached(priority: .userInitiated) { () -> CGImage? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixels,
            ]
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }.value
        if let image {
            cache.images.setObject(image, forKey: key, cost: image.bytesPerRow * image.height)
        }
        return image
    }

    /// Pixel size without decoding the image.
    static func pixelSize(of url: URL) -> CGSize? {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return CGSize(width: width, height: height)
    }

    /// Any image ImageIO reads (HEIC, RAW, TIFF, ...) as an upright PNG, which the engine accepts.
    ///
    /// The thumbnail path applies the EXIF orientation, as upstream's `exif_transpose` does;
    /// 4096 pixels keeps a camera photo's PNG under the engine's upload limit, and depth
    /// and upscaling look at no more than 1024 pixels of the long side by default.
    static func uploadPNG(from url: URL) throws -> Data {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 4096,
        ]
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return data as Data
    }
}
