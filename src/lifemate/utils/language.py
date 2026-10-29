"""Unchanged exploratory EDA regexes; these are not semantic annotations."""
import re
import unicodedata

PATTERNS = {
    'relative_time': r'\b(mai|mốt|hôm nay|tối nay|sáng nay|chiều nay|ngày kia|tuần sau|tháng sau|lát nữa|nữa)\b',
    'clock_time': r'\b(\d{1,2}\s*(giờ|h|phút)|sáng|chiều|tối|trưa|đêm)\b',
    'date': r'\b(ngày\s+\d+|tháng\s+\d+|thứ\s+(hai|ba|tư|năm|sáu|bảy)|chủ nhật|\d{1,2}[/-]\d{1,2})\b',
    'digit': r'\d',
    'quantity': r'\b(\d+\s*(cái|chiếc|lần|nghìn|ngàn|triệu|đồng|phút|giờ|độ|phần)|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b',
    'contact': r'\b(gọi|nhắn|liên hệ|danh bạ|số điện thoại|gửi.*cho)\b',
    'negation': r'\b(đừng|không|chớ|thôi|khỏi)\b',
    'correction': r'\b(nhầm|à không|ý là|đổi lại|sửa lại|thay vì)\b',
    'conjunction': r'\b(rồi|sau đó|và|tiếp theo|xong|đồng thời)\b',
    'reference': r'\b(nó|cái đó|người đó|lúc nãy|vừa rồi|đó|ấy|như cũ|tiếp|lại)\b',
    'english_indicator': r'\b(wifi|wi fi|bluetooth|email|mail|app|online|meeting|deadline|playlist|youtube|spotify|sms|tv|ok|okay|check|team|reminder)\b',
}

REGEX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}

def language_flags(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return {k: bool(p.search(text)) for k, p in REGEX.items()}
