from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
def autocruise(matrix):
    # 创建网格对象
    grid = Grid(matrix=matrix)

    # 设置起点和终点
    start = grid.node(0, 0)
    end = grid.node(9, 5)

    # 创建A*寻路器
    finder = AStarFinder()

    # 寻路并获取路径
    path, _ = finder.find_path(start, end, grid)
    
    return path
