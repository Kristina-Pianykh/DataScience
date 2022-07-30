from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union


class FormattedDate(datetime):
    def __new__(cls, year, month):
        return super().__new__(cls, year, month, 1, 0, 0, 0, 0, None)

    def __repr__(self):
        padding: str = "0" if self.month < 10 else ""
        return f"{self.year}-{padding}{self.month}"


def get_average(iterable: List[float]) -> float:
    """
    returns a mean value rounded to 3 digits after comma
    """
    average: float = sum(iterable) / float(len(iterable))
    return round(average, 3)


def read_entries(input_file: Path) -> List[Tuple]:
    """
    Reads the measured sunshine data from the file file into a tuple data structure.
    The selected tuple format defines the input data type of the following format:

    :param input_file: Path to the input file
    :return: all entries from the input file as a list of tuples
    """
    data: List[Tuple] = []
    MonthEntry = Tuple[FormattedDate, float]
    with open(input_file) as file:
        for row in file:
            date_, hours = row.strip().split(",")
            try:
                datetime_obj = datetime.strptime(date_, "%Y-%m")
                formatted_date: FormattedDate = FormattedDate(
                    datetime_obj.year, datetime_obj.month
                )
                MonthEntry = (formatted_date, float(hours))
                data.append(MonthEntry)
            except ValueError:
                continue
    return data


def get_average_sunshine_hours(entries: List[Tuple]) -> float:
    """
    Returns the average number of sunshine hours of all month entries of the entries
    (as float). The result is rounded to 3 decimal places.

    :param entries: List of monthly entries to be analyzed
    :return: Average number of sun hours of all entries
    """
    all_hours: List[float] = [entry[1] for entry in entries]
    return get_average(all_hours)


def number_of_months_with_min_sunshine_hours(
    entries: List[Tuple], num_hours: float
) -> int:
    """
    Returns the number of month entries in the entries list
    in which the average number of sun hours was at least num_hours.

    :param entries: list of monthly entries to be analyzed
    :param num_hours: Minimum number of sunshine hours
    :return: number of month entries in which the average number of sunshine hours was at least num_hours
    """
    months_more_sun: List[Tuple[datetime, float]] = []
    for entry in entries:
        try:
            if entry[1] >= num_hours:
                months_more_sun.append(entry)
        except TypeError:
            continue
    return len(months_more_sun)


def get_max_difference(entries: List[Tuple]) -> float:
    """
    Gibt die maximale Differenz zweier beliebiger Monatseinträge aus der Liste entries zurück. Die Differenz wird
    dabei auf drei Nachkommastellen gerundet.

    :param entries: Liste zu analysierende Monatseinträge
    :return: Maximale Differenz der durchschnittlichen Sonnenscheinstunden zweier beliebiger Monatseinträge
    """
    all_hours: List[float] = [entry[1] for entry in entries]
    max_hours: float = max(all_hours)
    min_hours: float = min(all_hours)
    return round(max_hours - min_hours, 3)


def get_average_sunshine_hours_by_year(
    entries: List[Tuple], year: int
) -> Union[float, None]:
    """
    Gibt die durchschnittliche Anzahl an Sonnenstunden aller in der Liste entries erfassten Einträge für das Jahre
    year zurück. Sollten keine Einträge in data für das Jahre year vorhanden sein, wird None zurückgegeben.
    Das Ergebnis wird auf drei Nachkommastellen gerundet.

    :param entries: Liste zu analysierende Monatseinträge
    :return: Durchschnittliche Anzahl an Sonnenstunden für das Jahr year
    """
    entries_per_year: List[float] = [row[1] for row in entries if row[0].year == year]
    if len(entries_per_year) == 0:
        return None
    else:
        return get_average(entries_per_year)


def get_average_sunshine_hours_by_month(entries: List[Tuple], month: str) -> float:
    """
    Gibt die durchschnittliche Anzahl an Sonnenstunden aller Einträge aus der Liste entries, d.h. über mehrere Jahre,
    für den Monat month zurück. Die Übergabe des Monats erfolgt als String und kann einen der folgenden Werte
    annehmen: “Jan”, “Feb”, “Mar”, “Apr”, “May”, “Jun”, “Jul”, “Aug”, “Sep”, “Oct”, “Nov”, “Dez”.
    Das Ergebnis wird auf drei Nachkommastellen gerundet.

    :param entries: Liste zu analysierende Monatseinträge
    :param month: Zu analysierender Monat
    :return Durchschnittliche Anzahl an Sonnenstunden für den Monat month (über mehrere Jahre)
    """
    months: List[str] = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dez",
    ]
    month_mapping: Dict[str, int] = {month: num for num, month in enumerate(months, 1)}
    entries_per_month: List[float] = [
        row[1] for row in entries if row[0].month == month_mapping[month]
    ]
    return get_average(entries_per_month)


def get_average_sunshine_hours_per_year(entries: List[Tuple]) -> Dict[int, float]:
    """
    Gibt die durchschnittliche Anzahl an Sonnenstunden aller in der Liste entries verständig erfassten Jahre zurück.
    Sollten für ein Jahr (noch) nicht alle Monate erfasst sein, wird dies Jahr ignoriert, d.h. kein Wert berechnet und
    zurückgeben. Die Rückgabe erfolgt als Dictionary, welches ein Jahr (als int) auf dessen die durchschnittliche
    Sonnenstundenanzahl (als float) abbildet.

    :param entries: Liste zu analysierende Monatseinträge
    :return: Dictionary mit den durchschnittlichen Sonnenstunden pro Jahr
    """
    # dictionary with a year as a key and a list of Nones/floats as values
    month_checker: Dict[int, List[Union[None, float]]] = {}

    for row in entries:
        try:
            # instantiate a list of 12 Nones (for each month) for each year in the data
            if row[0].year not in month_checker:
                month_checker[row[0].year] = [None] * 12
            # compute index based on the month value
            # (e.g. Jan is 1. => hence 1st element in the list)
            idx_by_month: int = row[0].month - 1
            month_checker[row[0].year][idx_by_month] = row[1]
        except AttributeError:
            continue
    # compute the mean hours of sunshine per year if there is a value for each month
    average_sunshine_per_year: Dict[int, float] = {
        year: get_average(month_checker[year])
        for year in month_checker
        if None not in month_checker[year]
    }
    return average_sunshine_per_year
