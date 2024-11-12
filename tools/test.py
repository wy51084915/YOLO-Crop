def get_geometric_series_weight(n):
    def sum_of_geometric_series(r, length):
        return (r ** length - 1) // (r - 1)

    max_weight = 0
    best_r = 0

    # 二分法寻找公比
    for r in range(2, n + 1):
        low, high = 1, n
        while low <= high:
            mid = (low + high) // 2
            if sum_of_geometric_series(r, mid) == n:
                if mid > max_weight:
                    max_weight = mid
                    best_r = r
                break
            elif sum_of_geometric_series(r, mid) < n:
                low = mid + 1
            else:
                high = mid - 1

    return max_weight, best_r

# 测试用例
n = 5
weight, ratio = get_geometric_series_weight(n)
print(weight, ratio)
