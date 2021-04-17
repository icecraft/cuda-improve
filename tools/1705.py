

import heapq
class Solution(object):
    def eatenApples(self, apples, days):
        """
        :type apples: List[int]
        :type days: List[int]
        :rtype: int
        """
        total, pos = 0, 1
        N, Q = len(apples), []
        
        for i in range(N):
            if apples[i] == 0:
                continue
            heapq.heappush(Q, [i+days[i]+1, apples[i], i+1])
        stack = []
        while Q:
            if pos >= Q[0][0]:
                stack.append(heapq.heappop(Q))
                continue
  
            rotten, count, j = Q[0]
            pos = max(pos, j)
            m = min(count, rotten - pos+1)
            if m > 0:
                pos += m
                total += m 
            print(Q[0], total, pos)
            stack.append(heapq.heappop(Q))
            
        print(stack)
        print(total)
    
    
    
if __name__ == '__main__':
    s = Solution()
    # s.eatenApples([2,1,1,4,5],[10,10,6,4,2])
    s.eatenApples([1,2,3,5,2],[3,2,1,4,2])