"""Stable errors exposed by the calendar application layer."""


class CalendarError(RuntimeError):
    """Base error for calendar operations."""


class CalendarValidationError(CalendarError, ValueError):
    """The requested operation violates a calendar business rule."""


class CalendarNotFoundError(CalendarError):
    """The requested calendar object is absent from the current chat."""


class CalendarConflictError(CalendarError):
    """The event was changed since the caller read its version."""
