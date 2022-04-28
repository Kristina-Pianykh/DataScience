import sys
from pathlib import Path

from analyze_sunshine_hours import (
    get_average_sunshine_hours,
    get_average_sunshine_hours_by_month,
    get_average_sunshine_hours_by_year,
    get_average_sunshine_hours_per_year,
    get_max_difference,
    number_of_months_with_min_sunshine_hours,
    read_entries,
)

if __name__ == "__main__":
    input_file = Path(sys.argv[1])

    # Pruefe, ob die Datei ueberhaupt existiert
    if not input_file.exists():
        # print(f'Eingabedatei {input_file} existiert nicht')
        exit(1)

    # Einlesen der Datei testen
    print("Teste Funktion read_data()")
    entries = read_entries(input_file)

    # Stelle sicher dass die korrekte Anzahl an Eintraegen eingelesen wurde
    expected_num_entries = 482
    if len(entries) == expected_num_entries:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours
    print("Teste Funktion get_average_sunshine_hours()")
    avg_hours = get_average_sunshine_hours(entries)

    expected_avg_hours = 5.687
    if avg_hours == expected_avg_hours:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours_by_month
    print("Teste Funktion number_of_months_with_min_sunshine_hours()")

    # Teste die Implementierung mittels Berechnung fuer 6,5 Stunden
    num_months_6_5 = number_of_months_with_min_sunshine_hours(entries, 6.5)
    expected_months_6_5 = 122

    if num_months_6_5 == expected_months_6_5:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # Teste die Implementierung mittels Berechnung fuer 25 Stunden (~ nicht moeglich)
    num_months_25 = number_of_months_with_min_sunshine_hours(entries, 25)
    expected_months_25 = 0

    if num_months_25 == expected_months_25:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours_by_month
    print("Teste Funktion get_max_difference()")

    max_diff = get_max_difference(entries)
    expected_max_diff = 6.6
    if max_diff == expected_max_diff:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours_by_month
    print("Teste Funktion get_average_sunshine_hours_by_year()")

    avg_1998 = get_average_sunshine_hours_by_year(entries, 1998)
    expected_1998_avg = 5.725
    if avg_1998 == expected_1998_avg:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    avg_1901 = get_average_sunshine_hours_by_year(entries, 1901)
    if avg_1901 is None:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours_by_month
    print("Teste Funktion get_average_sunshine_hours_by_month()")

    # Teste den Wert fuer Januar
    jan_avg = get_average_sunshine_hours_by_month(entries, "Jan")
    expected_jan_avg = 5.72
    if jan_avg == expected_jan_avg:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # Teste den Wert fuer Oktober
    oct_avg = get_average_sunshine_hours_by_month(entries, "Oct")
    expected_oct_avg = 5.062
    if oct_avg == expected_oct_avg:
        print("[PASS]")
    else:
        print("[FAIL]")
        exit(1)

    # ----------

    # Pruefe Funktion get_average_sunshine_hours_by_month
    print("Teste Funktion get_average_sunshine_hours_per_year()")
    year_to_avg = get_average_sunshine_hours_per_year(entries)
    expected_size = 40
    if len(year_to_avg) == expected_size:
        print("[PASS]")
    else:
        print("FAIL")

    print("Alle Tests bestanden!")
    exit(0)
