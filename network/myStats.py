import pstats
from pstats import SortKey


p = pstats.Stats('output')
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
