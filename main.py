import heapq
from typing import Dict, List, Tuple


class Graph:
    """
    Класс Graph представляет граф дорог между городами.

    Граф хранится в виде списка смежности:
    adj[u] = список кортежей (v, длина, время, стоимость),
    где u и v — идентификаторы городов.
    """

    def __init__(self):
        # Словарь: город -> список соседей с параметрами дороги
        self.adj: Dict[int, List[Tuple[int, int, int, int]]] = {}

    def add_edge(self, u: int, v: int, d: int, t: int, c: int):
        """
        Добавляет двустороннюю дорогу между городами u и v.

        :param u: ID первого города
        :param v: ID второго города
        :param d: длина дороги (км)
        :param t: время в пути (мин)
        :param c: стоимость проезда (руб)
        """
        # Так как дороги двусторонние — добавляем ребро в обе стороны
        self.adj.setdefault(u, []).append((v, d, t, c))
        self.adj.setdefault(v, []).append((u, d, t, c))


def dijkstra_single(graph: Graph, start: int, end: int, criterion: int):
    """
    Реализация алгоритма Дейкстры для оптимизации ОДНОГО критерия.

    criterion:
        0 — минимизация длины
        1 — минимизация времени
        2 — минимизация стоимости

    При этом параллельно суммируются все три параметра маршрута.

    :return:
        путь (список городов),
        суммарная длина,
        суммарное время,
        суммарная стоимость
    """

    # Очередь с приоритетом:
    # (значение оптимизируемого критерия, текущий город, D, T, C)
    pq = [(0, start, 0, 0, 0)]

    # dist[u] — минимальное найденное значение критерия для города u
    dist = {start: 0}

    # parent[v] = (u, D, T, C)
    # нужно для восстановления маршрута
    parent = {}

    while pq:
        cur, u, d_sum, t_sum, c_sum = heapq.heappop(pq)

        # Если дошли до конечного города — можно завершать
        if u == end:
            break

        # Если найден путь хуже уже известного — пропускаем
        if cur > dist.get(u, float('inf')):
            continue

        # Перебираем все дороги из текущего города
        for v, d, t, c in graph.adj.get(u, []):
            # Массив весов, чтобы удобно выбирать критерий
            weights = [d, t, c]

            # Новое значение оптимизируемого критерия
            new_cost = cur + weights[criterion]

            # Если нашли более выгодный путь
            if new_cost < dist.get(v, float('inf')):
                dist[v] = new_cost

                # Запоминаем, откуда пришли и накопленные параметры
                parent[v] = (u, d_sum + d, t_sum + t, c_sum + c)

                # Добавляем в очередь
                heapq.heappush(
                    pq,
                    (new_cost, v, d_sum + d, t_sum + t, c_sum + c)
                )

    # Восстанавливаем путь от end к start
    return reconstruct_path(parent, start, end)


def dijkstra_compromise(graph: Graph, start: int, end: int, priorities: List[int]):
    """
    Модифицированный алгоритм Дейкстры для компромиссного маршрута.

    Используется ЛЕКСИКОГРАФИЧЕСКАЯ оптимизация:
    сначала минимизируется самый важный критерий,
    затем второй по важности,
    затем третий.

    priorities — список индексов критериев в порядке важности
    (например, [0, 2, 1] означает Д → С → В)
    """

    # Очередь с приоритетом:
    # ((критерий1, критерий2, критерий3), город)
    pq = [((0, 0, 0), start)]

    # dist[u] — лучший кортеж критериев для города u
    dist = {start: (0, 0, 0)}

    # parent[v] = (u, d, t, c)
    parent = {}

    while pq:
        cur, u = heapq.heappop(pq)

        if u == end:
            break

        # Если текущее решение хуже найденного ранее — пропускаем
        if cur > dist.get(u, (float('inf'),) * 3):
            continue

        for v, d, t, c in graph.adj.get(u, []):
            values = [d, t, c]

            # Формируем новый кортеж критериев
            new = tuple(
                cur[i] + values[priorities[i]]
                for i in range(3)
            )

            # Лексикографическое сравнение кортежей
            if new < dist.get(v, (float('inf'),) * 3):
                dist[v] = new
                parent[v] = (u, d, t, c)
                heapq.heappush(pq, (new, v))

    return reconstruct_path(parent, start, end)


def reconstruct_path(parent, start, end):
    """
    Восстанавливает маршрут от start до end
    и подсчитывает суммарные параметры пути.
    """

    path = [end]
    d = t = c = 0

    # Двигаемся от конечного города к начальному
    while path[-1] != start:
        u, dd, tt, cc = parent[path[-1]]
        d += dd
        t += tt
        c += cc
        path.append(u)

    # Разворачиваем маршрут
    path.reverse()
    return path, d, t, c


def parse_input(filename: str):
    """
    Читает входной файл input.txt и разбирает его по секциям.

    :return:
        cities_id   — имя города -> ID
        cities_name — ID -> имя города
        roads       — список дорог
        requests    — список запросов
    """

    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    section = None
    cities_id = {}
    cities_name = {}
    roads = []
    requests = []

    for line in lines:
        # Определяем текущую секцию
        if line.startswith('['):
            section = line
            continue

        if section == '[CITIES]':
            cid, name = line.split(':', 1)
            cities_id[name.strip()] = int(cid)
            cities_name[int(cid)] = name.strip()

        elif section == '[ROADS]':
            left, right = line.split(':')
            u, v = map(int, left.split('-'))
            d, t, c = map(int, right.split(','))
            roads.append((u, v, d, t, c))

        elif section == '[REQUESTS]':
            route, pr = line.split('|')
            src, dst = map(str.strip, route.split('->'))
            priorities = pr.strip()[1:-1].split(',')
            requests.append((src, dst, priorities))

    return cities_id, cities_name, roads, requests


def main():
    """
    Основная функция программы.
    Управляет чтением данных, вычислением маршрутов и выводом.
    """

    cities_id, cities_name, roads, requests = parse_input("input.txt")

    # Создаём граф и добавляем все дороги
    graph = Graph()
    for u, v, d, t, c in roads:
        graph.add_edge(u, v, d, t, c)

    with open("output.txt", "w", encoding="utf-8") as out:
        for src, dst, pr in requests:
            start = cities_id[src]
            end = cities_id[dst]

            # Оптимальные маршруты по каждому критерию
            results = {
                "ДЛИНА": dijkstra_single(graph, start, end, 0),
                "ВРЕМЯ": dijkstra_single(graph, start, end, 1),
                "СТОИМОСТЬ": dijkstra_single(graph, start, end, 2),
            }

            for key, (path, d, t, c) in results.items():
                cities = " -> ".join(cities_name[x] for x in path)
                out.write(f"{key}: {cities} | Д={d}, В={t}, С={c}\n")

            # Преобразуем приоритеты в индексы
            priority_map = {'Д': 0, 'В': 1, 'С': 2}
            pr_idx = [priority_map[x] for x in pr]

            # Компромиссный маршрут
            path, d, t, c = dijkstra_compromise(graph, start, end, pr_idx)
            cities = " -> ".join(cities_name[x] for x in path)

            out.write(f"КОМПРОМИСС: {cities} | Д={d}, В={t}, С={c}\n\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("No input.txt file found.")