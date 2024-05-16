from datetime import datetime, timedelta

from .base_connector import BaseConnector


class BaseDateRangeConnector(BaseConnector):
    def __init__(self, start_date, end_date, **kwargs):
        self.start_date = start_date
        self.end_date = (
            end_date if end_date != "now" else datetime.now().strftime("%Y-%m-%d")
        )

    def calculate_week_ranges(self):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        delta = timedelta(days=1)
        while start < end:
            yield start.strftime("%Y%m%d")
            start += delta

    def calculate_day_ranges(self):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        delta = timedelta(days=1)
        while start < end:
            next_day = start + delta
            yield start.strftime("%Y%m%d"), next_day.strftime("%Y%m%d")
            start = next_day
