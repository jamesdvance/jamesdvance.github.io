---
layout: post
title: Charming Algorithms 1 - '3Sum Closest'
date: 2023-12-02 13:11:00
description: Intuition For the solution to '3Sum - Closest'
tags: algorithms, leetcode
categories: charming-algorithms
featured: true
---

## Charming Algorithms 1 : 3Sum (No, not that) Closest

### Problem
This [problem](https://leetcode.com/problems/3sum-closest) asks the user to find the three numbers in an array whose sum is closet to the target number. 

### Intuition 
The brute force method would be to multiply every 3 combinations of numbers by each other and finding the closet to the target value. This would have 'N choose K' time complexity where K = 3. n!/k!(n-k)!. We never want factorials to be part of our time complexity, so we have to look for a better solution. 

Similar to 2Sum and 3Sum, we can sort the array to reduce our exploration space. After sorting, we have a setup for a two-pointer technique. A minor detail is that we need to evaluate 3 points, not two. However, similar to the 3Sum solution, we can isolate a two-sum solution quite easily. To do so, we iterate over the array and create a subarray from our iteration point to the end of the array. In that subarray, our first pointer will always be the first element of the array. The other two pointers are assigned using the two-pointer technique. One starts at the first available starting position (start + 1) and the other at the end of the array. From there, the algorithm evaluates the sum of the 3 pointers. If the sum equals the target, we can return the sum. If it is closer than our existing solution, it takes the place of our existing solution. If neither is true, then we move the two pointers closer to each other. If the current sum is less than the target, then we want our next sum to be larger than this one. So we move the first of the two pointers up by one. If the current sum is already greater than the target, then we move the last of the pointers down by one. Because the array is sorted, these conditions make our decision process straightforward. Once the first and last of the two-pointers meet, we stop the algorithm. 

### Runtime
The solution becomes O(N^2) because in worst case we iterate over the array once and for each iteration we iterate over a subset of the entire array again. It's like a double for-loop where the second loop gets smaller each time. So technically its less than O(N^2) but for rule-of-thumb estimate, O(N^2) works fine. 

### Steps 
1. Sort the array
2. Initialize the result as the sum of the first 3 elements. 
3. Iterate over the array. At each iteration, perform the two-pointer iteration starting at the next index and ending at the end of the array. 
4. Return the result if the sum of the three indexes matches the target, re-assign if the new sum is closer than the old one, or increment or decrement the two-pointer locations depending on whether our current sum is bigger or smaller than the target. 

### Code 

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        i,j,k = 0,0,0

        nums.sort() # O(NlogN)

        result = sum(nums[:3])

        for i in range(len(nums)-2):
            j, k = i+1, len(nums) -1 

            while j < k:
                total = nums[i] + nums[j] + nums[k]

                if total == target:
                    return total
                
                if abs(total - target) < abs(result - target):
                    result = total 
                
                if total < target: 
                    j += 1
                
                else:
                    k -= 1
            
        return result
```
