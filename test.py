

def solution(A, B):
    answer = 0

    for a, b in zip(sorted(A), sorted(B, reverse=True)):
        answer += a*b

    return answer



if __name__ == "__main__":
    A = [1, 4, 2]
    B = [5, 4, 4]
    print(solution(A, B))
    