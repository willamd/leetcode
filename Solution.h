//
// Created by william on 19-5-29.
//

#ifndef ALG_SOLUTION_H
#define ALG_SOLUTION_H

#include <stdint.h>
#include <vector>
#include <string>

class Solution {

public:
    Solution();

    ~Solution();

    //成员声明函数
    int hammingWeight(uint32_t n);

    int hammingDistance(int x, int y);

    uint32_t reverseBits(uint32_t n);

    std::vector<std::vector<int>> generate(int numRows);

    int missingNumber(std::vector<int> &nums);

    bool isValid(std::string s);


    std::vector<int> reset();

    std::vector<int> shuffle();

    Solution(std::vector<int> &nums) : originNums(nums) {
    }

    int climbStairs(int n);

    int maxProfit(std::vector<int> &prices);

    int maxSubArray(std::vector<int> &nums);

    int rob(std::vector<int> &nums);

    void merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n);

    int firstBadVersion(int n);

    int lengthOfLongestSubstring(std::string s);

    bool checkInclusion(std::string s1, std::string s2);

    std::string multiply(std::string num1, std::string num2);

    std::string reverseWords(std::string s);

    std::string reverseWords1(std::string s);

    std::string simplifyPath(std::string path);

    std::vector<std::string> restoreIpAddresses(std::string s);


    std::string simplifyPath1(std::string path);

    std::vector<std::vector<int>> threeSum(std::vector<int> &nums);

    int maxAreaOfIsland(std::vector<std::vector<int>> &grid);

    int search(std::vector<int> &nums, int target);

    int findLengthOfLCIS(std::vector<int> &nums);

    int findKthLargest(std::vector<int> &nums, int k);

    int longestConsecutive(std::vector<int> &nums);

    std::string getPermutation(int n, int k);

    int trap(std::vector<int> &height);

    std::vector<std::vector<int>> merge(std::vector<std::vector<int>> &intervals);

    int findCircleNum(std::vector<std::vector<int>> &M);

    int findCircleNumBfs(std::vector<std::vector<int>> &M);

    int findCircleNumUnion(std::vector<std::vector<int>> &M);
private:
    std::vector<int> originNums;
    int x;
    int y;

};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(nums);
 * vector<int> param_1 = obj->reset();
 * vector<int> param_2 = obj->shuffle();
 */

#endif //ALG_SOLUTION_H
