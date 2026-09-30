"""A wrapping layout so no row of controls can ever overflow its panel.

The responsive contract of ASTRALINK is that the operator never has to
scroll sideways to reach a control.  A plain QHBoxLayout cannot honour that:
once the panel is narrower than the sum of its buttons the row simply pushes
the content wider than the viewport (the reported cut-off BENCHMARK VIDEO /
EXPORT REPORT / INSTANT REPORT / 3D DIGITAL TWIN launchers).

``FlowLayout`` keeps every item at its natural, readable size and moves the
ones that no longer fit onto the next row — buttons wrap instead of being
clipped, at any window width, with no horizontal scrollbar.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """QLayout that wraps its items onto new rows (right-to-left flow)."""

    def __init__(self, parent=None, margin=0, h_spacing=6, v_spacing=6):
        super().__init__(parent)
        self._items = []
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)
        self.setContentsMargins(margin, margin, margin, margin)

    # ---- QLayout plumbing ------------------------------------------------
    def addItem(self, item):  # noqa: N802 (Qt naming)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                      margins.top() + margins.bottom())
        return size

    # ---- wrapping engine -------------------------------------------------
    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(),
                                  -margins.right(), -margins.bottom())

        # 1. Split the items into the rows they wrap onto.
        rows = []
        row = []
        used = 0
        available = max(0, effective.width())
        for item in self._items:
            width = item.sizeHint().width()
            extra = width if not row else width + self._h_spacing
            if row and used + extra > available:
                rows.append((row, used))
                row, used = [], 0
                extra = width
            row.append(item)
            used += extra
        if row:
            rows.append((row, used))

        # 2. Place each row, sharing any leftover width between its items so a
        #    row still fills the panel exactly like the original grid did
        #    (never wider, never clipped).
        if not rows:
            return margins.top() + margins.bottom()
        y = effective.y()
        for row_items, used in rows:
            share_count = len(row_items)
            leftover = max(0, effective.width() - used)
            bonus = int(leftover / share_count) if share_count else 0
            x = effective.x()
            line_height = 0
            for index, item in enumerate(row_items):
                hint = item.sizeHint()
                width = hint.width()
                if bonus:
                    grow = leftover - bonus * (share_count - 1) if index == share_count - 1 else bonus
                    width = max(width, min(width + max(0, grow),
                                           item.maximumSize().width()))
                if not test_only:
                    item.setGeometry(QRect(QPoint(x, y), QSize(width, hint.height())))
                x += width + self._h_spacing
                line_height = max(line_height, hint.height())
            y += line_height + self._v_spacing
        return max(0, y - self._v_spacing - rect.y() + margins.bottom())
