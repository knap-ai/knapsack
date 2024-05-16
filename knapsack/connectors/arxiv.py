import pytz
import re
import typing as t
from datetime import datetime

import feedparser
import uuid
from tqdm import tqdm

from knapsack.connectors.base_date_range_connector import BaseDateRangeConnector


class Entry(object):
    """This class represents one arxiv entry"""

    def __init__(self, id: str, title: str,
                 authors: list, abstract: str, category: str = "",
                 date_submitted: datetime | None = None,
                 date_updated: datetime | None = None,
                 number: int | None = None):
        self.id = id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.category = category
        self.date_submitted = date_submitted
        self.date_updated = date_updated

        self.title_marks = []
        self.author_marks = [False] * len(self.authors)
        self.rating = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n    id={self.id!r},\n    title={self.title!r},\n    "
            f"authors={self.authors!r},\n    abstract={self.abstract!r},\n    "
            f"category={self.category!r},\n    "
            f"date_submitted={self.date_submitted!r},\n    date_updated={self.date_updated!r}"
            "\n)"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "category": self.category,
            "date_submitted": self.date_submitted,
            "date_updated": self.date_updated,
            "rating": self.rating,
        }

    def uuid(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.id))

    def to_embed(self) -> str:
        """Return a string representation of the entry for embedding"""
        return f"{self.title}\n{self.abstract}"

    def mark_title_position(self, position: int) -> None:
        """Mark title at given position"""
        self.title_marks.append(position)

    def mark_title_keyword(self, keyword: str) -> None:
        """Mark title at positions where keyword is found"""
        counts = self.title.lower().count(keyword)
        for _ in range(counts):
            starts = [m.start() for m in re.finditer(keyword, self.title.lower())]
            ends = [m.end() for m in re.finditer(keyword, self.title.lower())]
            for s, e in zip(starts, ends):
                for pos in range(s, e):
                    self.mark_title_position(pos)

    def mark_author(self, number: int) -> None:
        """Mark author (by given number in author list)"""
        self.author_marks[number] = True

    def evaluate(self, keyword_ratings: dict, author_ratings: dict,
                       rate_abstract: bool=True) -> int:
        """Evaluate entry

        Rate entries according to keywords and author list.
        This sets the rating attribute and marks title and marks title words and authors.

        Args:
            keywords (dict): dict with keywords as keys and rating as value
            authors (dict): dict with authors as keys and rating as value
        Returns:
            int: rating for this entry
        """
        self.rating = 0
        # find keywords in title and abstract
        for keyword, rating in keyword_ratings.items():
            keyword = keyword.lower()
            # find and mark keyword in title
            counts = self.title.lower().count(keyword)
            if counts > 0:
                self.mark_title_keyword(keyword)
                self.rating += counts * rating
            # find keyword in abstract
            if rate_abstract:
                self.rating += self.abstract.lower().count(keyword) * rating

        # find authors
        for author, rating in author_ratings.items():
            for i, a in enumerate(self.authors):
                match = re.search(r'\b{}\b'.format(author), a, flags=re.IGNORECASE)
                if match:
                    self.mark_author(i)
                    self.rating += rating

        return self.rating


def evaluate_entries(entries: list, keyword_ratings: dict, author_ratings: dict, rate_abstract: bool=True):
    """Evaluate all entries in list"""
    for entry in entries:
        entry.evaluate(keyword_ratings, author_ratings, rate_abstract)


def sort_entries(entries: list, rating_min: int, reverse: bool, length: int) -> list:
    ''' Sort entries by rating

    Only entries with rating >= rating_min are listed, and the list is at
    maximum length entries long. If reverse is True, the entries are reversed
    (after cutting the list to length entries). Note that the default order
    is most relevant paper on top.
    '''
    if length < 0:
        length = None

    # remove entries with low rating
    entries_filtered = filter(lambda entry: entry.rating >= rating_min, entries)
    # sort by rating
    results = sorted(entries_filtered, key=lambda x: x.rating, reverse=not reverse)

    return results[:length]


def linebreak_fix(text: str):
    """Replace linebreaks and indenting with single space"""
    return " ".join(line.strip() for line in text.split("\n"))


def datetime_fromisoformat(datestr: str):
    """Convert iso formatted datetime string to datetime object

    This is only needed for compatibility, as datetime.fromisoformat()
    was added in Python 3.7
    """
    return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S")


class ArXivConnector(BaseDateRangeConnector):
    def __init__(self, max_results: int | None = 1024, **kwargs):
        super().__init__(**kwargs)
        self.max_results = max_results

    def fetch(self):
        print(f"Fetching from {self.start_date} to {self.end_date}")
        base_url = "http://export.arxiv.org/api/query?"
        # TODO: generalize
        search_query = f'search_query=cat:cs.AI'

        for day_start, day_end in self.calculate_day_ranges():
            start: int = 0

            finished = False
            while not finished:

                time_range = f"submittedDate:{day_start}"

                query_url = f'{base_url}{search_query}&{time_range}&start={start}&max_results={self.max_results}'
                
                # Fetch the feed
                feed = feedparser.parse(query_url)
                if feed.bozo:
                    raise feed.bozo_exception
                if len(feed.entries) == 0:
                    break

                results = []
                with tqdm(total=len(feed.entries)) as pbar:
                    for feedentry in feed.entries:
                        entry = self.convert_to_entry(feedentry)
                        pbar.set_description(f"Processing: {entry.title}...")
                        pbar.update(1)
                        if (entry.date_submitted and 
                            entry.date_submitted > pytz.utc.localize(datetime.strptime(day_end, '%Y%m%d'))):
                            finished = True
                            break
                        results.append(entry)
                start += self.max_results
                yield results

    def knapsack_tags(self) -> dict[str, t.Any]:
        """
        Return the tags associated with data from this source.
        Used for ex: querying the VectorDB for results from this 
        connector.
        """
        return {
            "connector": "arxiv",
            "metadata": ["title", "category", "date_submitted", "abstract", "authors", "date_updated", "rating", "category"],
            "embed": ["title", "abstract"],
        }

    def convert_to_entry(self, entry) -> Entry:
        return Entry(
            id=entry.id.split("/")[-1],
            title=linebreak_fix(entry.title),
            authors=[author["name"] for author in entry.authors],
            abstract=linebreak_fix(entry.summary),
            category=entry.arxiv_primary_category["term"],
            date_submitted=pytz.utc.localize(datetime_fromisoformat(entry.published[:-1])),
            date_updated=pytz.utc.localize(datetime_fromisoformat(entry.updated[:-1])),
        )
