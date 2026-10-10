import Foundation

/// Conversations grouped by when they were last used, newest first, as the sidebar lists them.
struct RecentGroup: Identifiable, Equatable {
    let title: String
    let conversations: [Conversation]

    var id: String { title }

    static func group(_ conversations: [Conversation], now: Date = .now, calendar: Calendar = .current) -> [RecentGroup] {
        let today = calendar.startOfDay(for: now)
        var groups: [RecentGroup] = []
        for conversation in conversations.sorted(by: { $0.updated > $1.updated }) {
            let title = Self.title(for: conversation.updated, today: today, calendar: calendar)
            if groups.last?.title == title {
                groups[groups.count - 1] = RecentGroup(title: title, conversations: groups[groups.count - 1].conversations + [conversation])
            } else {
                groups.append(RecentGroup(title: title, conversations: [conversation]))
            }
        }
        return groups
    }

    private static func title(for date: Date, today: Date, calendar: Calendar) -> String {
        let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: date), to: today).day ?? 0
        switch days {
        case ...0: return String(localized: "Today")
        case 1: return String(localized: "Yesterday")
        case 2...7: return String(localized: "Previous 7 Days")
        case 8...30: return String(localized: "Previous 30 Days")
        default:
            let sameYear = calendar.component(.year, from: date) == calendar.component(.year, from: today)
            return date.formatted(sameYear ? .dateTime.month(.wide) : .dateTime.month(.wide).year())
        }
    }
}

extension Conversation {
    /// The newest finished result, whatever made it, for thumbnails.
    var latestResult: Entry? { entries.last(where: \.isDone) }
}
