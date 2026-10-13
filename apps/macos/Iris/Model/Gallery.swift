import Foundation

/// A result as the gallery lists it, with where it is kept.
struct GalleryItem: Identifiable, Equatable {
    let entry: Entry
    let conversation: UUID
    let project: UUID?

    var id: UUID { entry.id }
}

/// The results of one day.
struct GallerySection: Identifiable, Equatable {
    let title: String
    var items: [GalleryItem]

    var id: String { title }
}

/// What the gallery shows, and in what order: every result was made on its own, so they are
/// listed newest first, with the images of one batch kept in the order they were made.
enum Gallery {
    /// The entries of a batch are created together, within microseconds; the next request
    /// comes after the previous one has finished, so a shorter gap means the same batch.
    static let batchGap: TimeInterval = 0.1

    static func items(
        of conversations: [Conversation], in scope: Workspace.Selection, matching query: String = ""
    ) -> [GalleryItem] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        // in the order they were made, wherever they are kept, so an image moved out of
        // its batch still sorts with it
        let made = conversations.filter(scope.contains)
            .flatMap { c in c.entries.map { GalleryItem(entry: $0, conversation: c.id, project: c.projectID) } }
            .enumerated()
            .sorted { ($0.element.entry.created, $0.offset) < ($1.element.entry.created, $1.offset) }
            .map(\.element)
        var keyed: [(batch: Date, order: Int, item: GalleryItem)] = []
        var batch = Date.distantPast
        var previous: Date?
        for (order, item) in made.enumerated() {
            if previous.map({ item.entry.created.timeIntervalSince($0) >= batchGap }) ?? true { batch = item.entry.created }
            previous = item.entry.created
            if query.isEmpty || item.entry.matches(query) { keyed.append((batch, order, item)) }
        }
        return keyed
            .sorted { $0.batch != $1.batch ? $0.batch > $1.batch : $0.order < $1.order }
            .map(\.item)
    }

    /// `items` in order, under "Today", "Yesterday" or their date.
    static func sections(_ items: [GalleryItem], now: Date = .now, calendar: Calendar = .current) -> [GallerySection] {
        let today = calendar.startOfDay(for: now)
        var sections: [GallerySection] = []
        for item in items {
            let title = dayTitle(item.entry.created, today: today, calendar: calendar)
            if sections.last?.title == title {
                sections[sections.count - 1].items.append(item)
            } else {
                sections.append(GallerySection(title: title, items: [item]))
            }
        }
        return sections
    }

    private static func dayTitle(_ date: Date, today: Date, calendar: Calendar) -> String {
        let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: date), to: today).day ?? 0
        switch days {
        case ...0: return String(localized: "Today")
        case 1: return String(localized: "Yesterday")
        default:
            let style = Date.FormatStyle(calendar: calendar, timeZone: calendar.timeZone).weekday(.wide).month(.wide).day()
            let sameYear = calendar.component(.year, from: date) == calendar.component(.year, from: today)
            return date.formatted(sameYear ? style : style.year())
        }
    }

    enum Move {
        case left, right, up, down
    }

    /// Where an arrow key takes the selection: along the order, or to the same column of the
    /// row above or below, each day being its own grid of `columns`. With nothing selected, the first item.
    static func neighbor(of id: UUID?, in sections: [GallerySection], columns: Int, moving move: Move) -> UUID? {
        let columns = max(columns, 1)
        let rows = sections.flatMap { section in
            stride(from: 0, to: section.items.count, by: columns).map { start in
                section.items[start..<min(start + columns, section.items.count)].map(\.id)
            }
        }
        guard let id, let row = rows.firstIndex(where: { $0.contains(id) }), let column = rows[row].firstIndex(of: id) else {
            return rows.first?.first
        }
        switch move {
        case .left, .right:
            let order = rows.flatMap { $0 }
            guard let index = order.firstIndex(of: id) else { return id }
            return order[(move == .left ? index - 1 : index + 1).clamped(to: 0...(order.count - 1))]
        case .up:
            guard row > 0 else { return id }
            return rows[row - 1][min(column, rows[row - 1].count - 1)]
        case .down:
            guard row < rows.count - 1 else { return id }
            return rows[row + 1][min(column, rows[row + 1].count - 1)]
        }
    }
}
