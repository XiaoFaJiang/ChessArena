from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        a = b = c = 0
        for x in nums:
            print(a,b,c)
            a,b,c = (a ^ x) & ((b & c & ~(a)) | (~b & ~c & a)),(b ^ x) & (c),(c ^ x) & (~a)
        print(a,b,c)
        return c


nums = [2,2,2,2,2,3]
a = Solution()
a.singleNumber(nums)