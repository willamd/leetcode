//
// Created by william on 19-5-29.
//

#include "Solution.h"
#include <stack>
#include <algorithm>
#include <chrono>
#include <random>
#include <queue>
#include <string>
#include <map>
#include <sstream>

using namespace std;

Solution::Solution() {

}

Solution::~Solution() {

}

int Solution::hammingWeight(uint32_t n) {
    int ret = 0;
    while (n) {
        if (n & 1 == 1) {
            ret++;
        }
        n = n >> 1;
    }
    return ret;
}

int Solution::hammingDistance(int x, int y) {
    int z = x ^y;
    int ret = 0;
    while (z) {
        if (z & 1 == 1) {
            ret++;
        }
        z = z >> 1;
    }
    return ret;
}

//颠倒给定的 32 位无符号整数的二进制位。10->01
uint32_t Solution::reverseBits(uint32_t n) {
    uint32_t ret = 0;
    for (int i = 0; i < 32; ++i) {
        ret <<= 1;
        ret = ret | (n & 1);
        n >>= 1;
    }
    return ret;
}

std::vector<std::vector<int>> Solution::generate(int numRows) {
    std::vector<std::vector<int>> ret(numRows, std::vector<int>());

    for (int i = 0; i < numRows; ++i) {
        ret[i].resize(i + 1, 1);
        for (int j = 1; j <= i - 1; ++j) {
            ret[i][j] = ret[i - 1][j - 1] + ret[i - 1][j];
        }

    }
    return ret;
}

int Solution::missingNumber(std::vector<int> &nums) {
    int vectorSize = nums.size();
    int ret = vectorSize * (vectorSize + 1) / 2;
    for (int i = 0; i < vectorSize; ++i) {
        ret -= nums[i];
    }
    return ret;
}

bool Solution::isValid(std::string s) {
    std::stack<char> ret;
    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == '(' || s[i] == '{' || s[i] == '[') {
            ret.push(s[i]);
        } else if (!ret.empty() && (ret.top() == '(' && s[i] == ')'
                                    || ret.top() == '[' && s[i] == ']'
                                    || ret.top() == '{' && s[i] == '}')) {
            ret.pop();
        } else {
            return false;
        }
    }
    if (ret.empty()) {
        return true;
    }
    return false;
}


/** Resets the array to its original configuration and return it. */
std::vector<int> Solution::reset() {
    return this->originNums;
}

/** Returns a random shuffling of the array. */
std::vector<int> Solution::shuffle() {
    std::vector<int> tmp(this->originNums);
    unsigned len = tmp.size();
    for (int i = len - 1; i >= 0; ++i) {
        std::swap(tmp[i], tmp[rand() % (i + 1)]);
//        std::swap(tmp[i], tmp[i + rand() % (tmp.size() - i)]);
    }
    return tmp;
}

//爬楼梯
int Solution::climbStairs(int n) {
    std::vector<int> ret(n + 1);
    if (n == 0) {
        return 1;
    }
    ret[0] = 1;
    ret[1] = 1;
//    return climbStairs(n - 1) + climbStairs(n - 2);
    for (int i = 2; i <= n; ++i) {
        ret[i] = ret[i - 1] + ret[i - 2];
    }
    return ret[n];
}

int Solution::maxProfit(std::vector<int> &prices) {
    int lenPrices = prices.size();
    int max = 0, min = prices[0];
    for (int i = 1; i < lenPrices; ++i) {
        min = std::min(prices[i], min);
        max = std::max(max, prices[i] - min);
    }
    return max;
}

int Solution::maxSubArray(std::vector<int> &nums) {
    int numSize = nums.size();
    //以Ai为结尾的最大子序列和
    std::vector<int> dp(numSize);
    dp[0] = nums[0];
    for (int i = 1; i < numSize; ++i) {
        dp[i] = std::max(dp[i - 1] + nums[i], nums[i]);
    }
    int ret = dp[0];
    for (int j = 1; j < dp.size(); ++j) {
        if (ret < dp[j])
            ret = dp[j];
    }
    return ret;
}

int Solution::rob(std::vector<int> &nums) {
//    int even = 0;
//    int old = 0;
//    for (int i = 0; i < nums.size(); ++i) {
//        if (i % 2 == 0) {
//            even =std::max(nums[i]+even,old);
//        } else {
//            old = std::max(nums[i]+old,even);
//        }
//
//    }
//    return std::max(even, old);
    int rob, notRob = 0;
    for (int i = 0; i < nums.size(); ++i) {
        int prerob = rob, notprerob = notRob;
        rob = notprerob + nums[i];
        notRob = std::max(prerob, notprerob);

    }
    return std::max(rob, notRob);
}

//给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
//
//说明:
//
//初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
//你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

void Solution::merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n) {
//    for (int i = m; i < m + n; ++i) {
//        nums1[i] = nums2[i - m];
//    }
    //std::sort(nums1.begin(), nums1.end());
    //有序数组,可以从后往前以次加入进去
    int totalNums = m + n;
    while (m > 0 && n > 0) {
        if (nums1[m - 1] >= nums2[n - 1]) {
            nums1[totalNums - 1] = nums1[m - 1];
            m--;
            totalNums--;
        } else {
            nums1[totalNums - 1] = nums2[n - 1];
            totalNums--;
            n--;
        }
    }
    //如果m>0的话,不用管保持原序,如果n>0;需要将n替换到num1
    while (n > 0) {
        nums1[totalNums - 1] = nums2[n - 1];
        totalNums--;
        n--;
    }
}

bool isBadVersion(int version) {

}


int Solution::firstBadVersion(int n) {
    int end = n;
    int start = 1;
    int mid = 0;
    if (isBadVersion(1)) {
        return 1;
    }
    while (start < end) {
        mid = start / 2 + end / 2;
        if (isBadVersion(mid)) {
            end = mid;
            continue;
        } else {
            start = mid + 1;
        }
    }
    return start;
}

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

int maxDepth(TreeNode *root) {
    if (root == NULL)
        return 0;
    if (root->left == NULL) {

        return maxDepth(root->right) + 1;
    } else if (root->right == NULL) {
        return maxDepth(root->left) + 1;
    } else {
        return std::max(1 + maxDepth(root->left), 1 + maxDepth(root->right));
    }
}

bool isValidBSt(TreeNode *root, long min, long max) {

    if (!root)
        return true;
    if (root->val >= max || root->val <= min)
        return false;
    return isValidBSt(root->left, min, root->val) && isValidBSt(root->right, root->val, max);
}

bool isValidBST(TreeNode *root) {
    return isValidBSt(root, INT64_MIN, INT64_MAX);
}

bool isValidBSTStack(TreeNode *root) {
    if (!root)
        return true;
    std::stack<TreeNode *> stk;
    TreeNode *pre = NULL;
    while (root || !stk.empty()) {
        if (root) {
            stk.push(root);
            root = root->left;
        } else {
            TreeNode *temp = stk.top();
            stk.pop();
            if (pre != NULL && temp->val <= pre->val) {
                return false;
            }
            pre = temp;
            root = temp->right;
        }
    }
    return true;
}

bool isEqual(TreeNode *ln, TreeNode *rn) {
    if (!ln && !rn) {
        return true;
    } else if (ln && rn) {
        return ln->val == rn->val && isEqual(ln->left, rn->right) && isEqual(ln->right, rn->left);
    } else {
        return false;
    }
}

bool isSymmetric(TreeNode *root) {
    if (!root)
        return true;

    return isEqual(root->left, root->right);
}

std::vector<std::vector<int>> levelOrder(TreeNode *root) {
    std::vector<std::vector<int >> ret;
    if (!root) {
        return ret;
    }
    std::queue<TreeNode *> stk;
    TreeNode *temp = NULL;
    stk.push(root);
    while (!stk.empty()) {
        std::vector<int> levelVec;
        int stkSize = stk.size();
        for (int i = 0; i < stkSize; ++i) {
            temp = stk.front();
            stk.pop();
            levelVec.push_back(temp->val);
            if (temp->left) stk.push(temp->left);
            if (temp->right) stk.push(temp->right);
        }
        ret.push_back(levelVec);
    }
    return ret;

}

void levelRecuresice(TreeNode *root, int level, std::vector<std::vector<int>> &res) {
    if (!root)
        return;

    if (level == res.size()) res.push_back({});
    res[level].push_back(root->val);
    if (root->left) levelRecuresice(root->left, level + 1, res);
    if (root->right) levelRecuresice(root->right, level + 1, res);

}

std::vector<std::vector<int>> levelOrderdg(TreeNode *root) {
    std::vector<std::vector<int>> res;
    levelRecuresice(root, 0, res);
    return std::vector<std::vector<int>>(res.rbegin(), res.rend());
}

TreeNode *sortedArrayToBST(std::vector<int> &nums, int left, int right) {
    //二分查找思想,二叉搜索树中序遍历完就是这个数组方式
    if (left > right) return NULL;
    int mid = (left + right) / 2;
    TreeNode *cur = new TreeNode(nums[mid]);
    cur->left = sortedArrayToBST(nums, left, mid - 1);
    cur->right = sortedArrayToBST(nums, mid + 1, right);
    return cur;
}

TreeNode *sortedArrayToBST(std::vector<int> &nums) {
    //二分查找思想
    return sortedArrayToBST(nums, 0, nums.size() - 1);
}

int Solution::lengthOfLongestSubstring(std::string s) {
    int subMap[255] = {0};
    int maxLen = 0, leftEle = 0;
    for (int i = 0; i < s.size(); ++i) {
        if (subMap[s[i]] == 0 || subMap[s[i]] < leftEle) {
            maxLen = std::max(i - leftEle + 1, maxLen);
        } else {
            leftEle = subMap[s[i]];
        }
        subMap[s[i]] = i + 1;

    }
    return maxLen;
}

bool isSame(int count[]) {
    for (int i = 0; i < 26; i++) {
        if (count[i] != 0) {
            return false;
        }
    }
    return true;
}

bool Solution::checkInclusion(std::string s1, std::string s2) {
    int s1Len = s1.length();
    int s2Len = s2.length();
    if (s1Len > s2Len)
        return false;

    int charMap[26] = {0};

    for (int i = 0; i < s1Len; ++i) {
        charMap[s1[i] - 'a']++;
        charMap[s2[i] - 'a']--;
    }
    if (isSame(charMap)) {
        return true;
    }
    for (int j = s1Len; j < s2Len; ++j) {
        charMap[s2[j] - 'a']--;//删除窗口首元素，移动窗口
        charMap[s2[j - s1Len] - 'a']++;
        if (isSame(charMap)) {
            return true;
        }
    }
    return false;
}

std::string Solution::multiply(std::string num1, std::string num2) {
    int s1Len = num1.length();
    int s2Len = num2.length();
    std::string res = "";
    int multiFlag = 0;
    int total = 0;
    std::vector<int> resData(s1Len + s2Len, 0);
    //错位相乘,最高位暂时保留
    for (int i = 0; i < s1Len; ++i) {
        for (int j = 0; j < s2Len; ++j) {
            resData[s1Len + s2Len - i - j - 2] += (num1[i] - '0') * (num2[j] - '0');

        }
    }
    for (int k = 0; k <= s1Len + s2Len - 1; k++) {
        total += resData[k];
        resData[k] += multiFlag;
        multiFlag = resData[k] / 10;
        resData[k] %= 10;
    }
    if (total == 0)
        return "0";
    for (int l = s1Len + s2Len - 1; l >= 0; l--) {
        if (l == s1Len + s2Len - 1 && resData[l] == 0) {
            continue;//过滤最高位为0
        }
        res.push_back(resData[l] + '0');
    }
    return res;
}

//方法1先翻转字符串然后再翻转单个单词
std::string Solution::reverseWords(std::string s) {
    std::string res = "";
    std::reverse(s.begin(), s.end());
    int sLen = s.length(), storeIndex = 0;
    for (int i = 0; i < sLen; ++i) {
        if (s[i] != ' ') {
            //首单词
            if (storeIndex != 0) s[storeIndex++] = ' ';
            int j = i;
            //下一个单词
            while (j < sLen && s[j] != ' ')
                s[storeIndex++] = s[j++];
            //storeIndex不为‘ ’ 的索引
            std::reverse(s.begin() + storeIndex - (j - i), s.begin() + storeIndex);
            i = j;
        }

    }
    s.resize(storeIndex);
    return s;
}

//isStream 稍微慢
std::string Solution::reverseWords1(std::string s) {
    std::stringstream istream(s);
    std::string temp = "";
    //读取第一个
    istream >> s;
    //istream 会提取空格作为分割
    while (istream >> temp)
        s = temp + " " + s;
    if (!s.empty() && s[0] == ' ') return "";
    return s;
}


//bool isValid(string s) {
//    if (s.empty() || s.size() > 3 || (s.size() > 1 && s[0] == '0')) return false;
//    int res = atoi(s.c_str());
//    return res <= 255 && res >= 0;
//}

//n_th 表示确定了几段
void helper(string s, int n_th, string out, vector<string> &res) {
    if (n_th == 4) {
        if (s.empty()) res.push_back(out);
    } else {
        //每一段1-3位都尝试
        for (int i = 1; i < 4; ++i) {
            //不够分的了
            if (s.size() < i) break;
            int val = atoi(s.substr(0, i).c_str());
            //过滤010这种
            if (val > 255 || i != to_string(val).size()) continue;
            helper(s.substr(i), n_th + 1, out + s.substr(0, i) + (n_th == 3 ? "" : "."), res);
        }
    }

}

vector<string> Solution::restoreIpAddresses(std::string s) {
    std::vector<std::string> res;
    helper(s, 0, "", res);
    return res;
}

string Solution::simplifyPath(string path) {
    string res, t;
    stringstream ss(path);
    vector<string> v;
    while (getline(ss, t, '/')) {
        if (t == "" || t == ".") continue;
        if (t == ".." && !v.empty()) v.pop_back();
        else if (t != "..") v.push_back(t);
    }
    for (string s : v) res += "/" + s;
    return res.empty() ? "/" : res;
}

string Solution::simplifyPath1(string path) {
    vector<string> v;
    int i = 0;
    while (i < path.size()) {
        while (path[i] == '/' && i < path.size()) ++i;
        if (i == path.size()) break;
        int start = i;
        while (path[i] != '/' && i < path.size()) ++i;
        int end = i - 1;
        string s = path.substr(start, end - start + 1);
        if (s == "..") {
            if (!v.empty()) v.pop_back();
        } else if (s != ".") {
            v.push_back(s);
        }
    }
    if (v.empty()) return "/";
    string res;
    for (int i = 0; i < v.size(); ++i) {
        res += '/' + v[i];
    }
    return res;
}

//a+b+c=0,绝对有负数，000
vector<vector<int>> Solution::threeSum(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> ret;
    map<int, int> elemMap;
    int numsLen = nums.size();
    for (int i = 0; i < numsLen; ++i) {
        //相等数字记录一次
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        int start = i + 1;
        int end = numsLen - 1;
        while (start < end) {
            if (nums[i] + nums[start] + nums[end] < 0)
                start++;
            else if (nums[i] + nums[start] + nums[end] > 0)
                end--;
            else {
                ret.push_back({nums[i], nums[start], nums[end]});
                //去重
                while (start < end && nums[start] == nums[start + 1]) start++;
                while (start < end && nums[end] == nums[end - 1])end--;
                start++;
                end--;
            }
        }
    }
    return ret;

}

//递归，已经访问的记录为0，深度搜索
int dsf(vector<vector<int>> &grid, int h, int w) {
    if (h >= 0 && h < grid.size() && w >= 0 && w < grid[0].size() && grid[h][w] == 1) {
        grid[h][w] = 0;
        return 1 + dsf(grid, h - 1, w) + dsf(grid, h + 1, w) + dsf(grid, h, w - 1) + dsf(grid, h, w + 1);
    } else {
        return 0;
    }
}

int Solution::maxAreaOfIsland(vector<vector<int>> &grid) {
    int gridLen = grid.size();
    int gridWth = 0;
    int maxArea = 0;
    if (gridLen > 0) {
        gridWth = grid[0].size();
    } else {
        return 0;
    }
    for (int i = 0; i < gridLen; ++i) {
        for (int j = 0; j < gridWth; ++j) {
            if (grid[i][j] != 0)
                maxArea = max(maxArea, dsf(grid, i, j));
        }
    }
    return maxArea;
}

//搜索旋转排序数组,中间数跟两边数比
int Solution::search(vector<int> &nums, int target) {
    int start = 0;
    int numsLen = nums.size();
    int end = numsLen - 1;
    int mid = 0;
    while (start <= end) {
        mid = (start + end) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < nums[end]) {
            //mid-end 升序
            if (nums[mid] < target && nums[end] >= target) start = mid + 1;
            else end = mid - 1;
        } else {
            //start-mid 升序
            if (nums[mid] > target && nums[start] <= target) end = mid - 1;
            else start = mid + 1;
        }
    }
    return -1;

}

//给定一个未经排序的整数数组，找到最长且连续的的递增序列。
int Solution::findLengthOfLCIS(vector<int> &nums) {
    int maxN = 1;
    int numLen = nums.size();
    if (numLen == 1) {
        return maxN;
    } else if (numLen < 1)
        return 0;
    int cnt = 1;
    for (int i = 1; i < numLen; ++i) {
        if (nums[i] > nums[i - 1])
            cnt++;
        else {
            cnt = 1;
        }
        maxN = max(cnt, maxN);
    }
    return maxN;
}

//数组中的第K个最大元素
int Solution::findKthLargest(vector<int> &nums, int k) {
    int numsLen = nums.size();
    sort(nums.begin(), nums.end());
    return nums[numsLen - k];
}

//最长连续序列
int Solution::longestConsecutive(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    int numsLen = nums.size();
    int maxL = 1;
    int cnt = 1;
    if (numsLen == 0)
        return 0;

    for (int i = 1; i < numsLen; ++i) {
        if (nums[i] - nums[i - 1] == 1)
            cnt++;
        else if (nums[i] == nums[i - 1]) continue;
        else
            cnt = 1;
        maxL = max(maxL, cnt);
    }
    return maxL;
}

//第k个排列
//next_permutation
//n的阶乘
//第几个数字开始的，然后取第几个值，逐步把数字从其中拿出
string Solution::getPermutation(int n, int k) {
    vector<int> fac(n + 1, 1);
    for (int i = 2; i <= n; ++i) {
        fac[i] = i * fac[i - 1];
    }
    vector<int> nums;
    for (int j = 1; j <= n; ++j) {
        nums.emplace_back(j);
    }
    string res("");
    for (int l = 1; l <= n; ++l) {
        //找开始点
        int idx = k / fac[n - l];
        //非整除+1在下一个数开始
        if (k % fac[n - l] != 0)
            idx++;
        res += (nums[idx - 1] + '0');
        nums.erase(nums.begin() + idx - 1);
        k = k - ((idx - 1) * fac[n - l]);
    }
    return res;
}

int Solution::trap(vector<int> &height) {
    int size = height.size();

    //从左到右搜索，分别记录左边与右边最高的。然后向较低的bin中注入水维持与右边同样高，注入水的和就是所求。
    int left = 0;
    int right = size - 1;
    int res = 0;
    int maxleft = 0, maxright = 0;
    while (left <= right) {
        if (height[left] <= height[right]) {
            if (height[left] >= maxleft) maxleft = height[left];
            else res += maxleft - height[left];
            left++;
        } else {
            if (height[right] >= maxright) maxright = height[right];
            else res += maxright - height[right];
            right--;
        }
    }
    return res;
}
//int n = height.size();
//// left[i]表示i左边的最大值，right[i]表示i右边的最大值
//vector<int> left(n), right(n);
//for (int i = 1; i < n; i++) {
//left[i] = max(left[i - 1], height[i - 1]);
//}
//for (int i = n - 2; i >= 0; i--) {
//right[i] = max(right[i + 1], height[i + 1]);
//}
//int water = 0;
//for (int i = 0; i < n; i++) {
//int level = min(left[i], right[i]);
//water += max(0, level - height[i]);
//}
//return water;

vector<vector<int>> Solution::merge(vector<vector<int>> &intervals) {
    int numsLen = intervals.size();
    if (numsLen <= 1)
        return intervals;
    vector<vector<int>> ret;
    vector<int> start(numsLen, 0);
    vector<int> end(numsLen, 0);
    for (int i = 0; i < numsLen; ++i) {
        start[i] = (intervals[i][0]);
        end[i] = (intervals[i][1]);
    }
    int idx = 0;
    sort(start.begin(), start.end());
    sort(end.begin(), end.end());
    //前一个元素的end与后一个元素的start比较，start大，合并前面的
    for (int j = 0; j < numsLen; ++j) {
        if (j < numsLen - 1 && start[j + 1] > end[j]) {
            ret.push_back({start[idx], end[j]});
            idx = j + 1;
        } else if (j == numsLen - 1) {
            ret.push_back({start[idx], end[j]});
        }
    }
    return ret;
}

//班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊
// 具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么
// 我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。
//深度，广度，差并集
void dfs(vector<vector<int>> &M, vector<bool> &visitor, int ith) {
    visitor[ith] = true;
    for (int j = 0; j < M.size(); ++j) {
        if (!visitor[j] && M[ith][j]) {
            dfs(M, visitor, j);
        }
    }
}


int Solution::findCircleNum(vector<vector<int>> &M) {
    int Mlen = M.size();
    vector<bool> visitor(Mlen, false);
    int cnt = 0;
    for (int i = 0; i < Mlen; ++i) {
        if (visitor[i]) continue;
        dfs(M, visitor, i);
        cnt++;
    }
    return cnt;
}

//广度
int Solution::findCircleNumBfs(vector<vector<int>> &M) {
    int Mlen = M.size();
    int cnt = 0;
    vector<bool> visitor(Mlen, false);
    queue<int> q;
    for (int i = 0; i < Mlen; ++i) {
        if (visitor[i]) continue;
        q.push(i);
        while (!q.empty()) {
            int front = q.front();
            q.pop();
            visitor[front] = true;
            for (int j = 0; j < Mlen; ++j) {
                if (M[i][j] && !visitor[j])
                    q.push(j);
            }
        }
        cnt++;
    }

    return cnt;
}

//差并集
//root存的是属于同一组的另一个对象的坐标，这样通过getRoot函数可以使同一个组的对象返回相同的值
int getParent(vector<int> &root, int ith) {
    while (ith != root[ith]) {
        //更新根节点，属于同一个集合
        root[ith] = root[root[ith]];
        ith = root[ith];
    }
    return ith;
}

int Solution::findCircleNumUnion(std::vector<std::vector<int>> &M) {
    int Mlen = M.size();
    int cnt = Mlen;
    vector<int> visitor(Mlen);
    for (int i = 0; i < Mlen; ++i) visitor[i] = i;
    for (int i = 0; i < Mlen; ++i) {
        for (int j = i + 1; j < Mlen; ++j) {
            if (M[i][j] == 1) {
                int paraent1 = getParent(visitor, i);
                int paraent2 = getParent(visitor, j);
                if (paraent1 != paraent2) {
                    --cnt;
                    visitor[paraent2] = paraent1;
                }
            }
        }
    }
    return cnt;
}

struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {}
};

ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    ListNode *ret = new ListNode(0);
    ListNode *cureent = ret;
    int flag = 0;
    while (l1 != NULL || l2 != NULL) {
        int l1Val = l1 != NULL ? l1->val : 0;
        int l2Val = l2 != NULL ? l2->val : 0;
        int sum = (l1Val + l2Val + flag);
        flag = sum / 10;
        cureent->next = new ListNode(sum % 10);
        cureent = cureent->next;
        if (l1 != NULL) l1 = l1->next;
        if (l2 != NULL) l2 = l2->next;
    }
    if (flag > 0)
        cureent->next = new ListNode(flag);
    return ret->next;

}

ListNode *merge(ListNode *l1, ListNode *l2) {
    ListNode *ret = new ListNode(0);
    ListNode *curr = ret;
    while (l1 && l2) {
        if (l1->val < l2->val) {
            curr->next = l1;
            curr = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            curr = l2;
            l2 = l2->next;
        }
    }
    if (l1)curr->next = l1;
    if (l2)curr->next = l2;
    return ret->next;
}

ListNode *sortList(ListNode *head) {
    if (!head || !head->next) return head;
    ListNode *pre = head;
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        pre = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    pre->next = NULL;//分2段

    return merge(sortList(head), sortList(slow));

}

ListNode *detectCycle(ListNode *head) {
    if (!head)return NULL;
    ListNode *slow = head;
    ListNode *fast = head;
    ListNode *entryNode = head;
    bool cycle = false;
    //在进入点相遇，说明在环内走了
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            cycle = true;
            //相遇后，慢指针围环走n+1圈后，+x后在进入环点相遇
            while (entryNode != slow) {
                entryNode = entryNode->next;
                slow = slow->next;
            }
            break;
        }
    }
    return cycle ? entryNode : NULL;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    //计算AB的长度，A=x,B=y,长的先走|x-y|后一起走绝对会相遇
    int LenA = 0;
    int LenB = 0;
    ListNode *a = headA;
    ListNode *b = headB;

    while (a) {
        a = a->next;
        LenA++;
    }
    while (b) {
        b = b->next;
        LenB++;
    }
    int diff = LenA - LenB;
    if (diff > 0)//说明A长，A先走diff步，否则b先走
    {
        a = headA;
        b = headB;
    } else {
        a = headB;
        b = headA;
        diff *= -1;
    }
    while (diff--) {
        a = a->next;
    }
    while (a != b) {
        a = a->next;
        b = b->next;
    }
    return a;
}

//每一列找最小的排序、或者转成vector后排序，然后转成链表
//归并

ListNode *mergeKLists(vector<ListNode *> &lists) {
    int listSize = lists.size();
    vector<int> tempV;
    ListNode *ret = new ListNode(0);
    ListNode *tempNode = ret;
    for (int i = 0; i < listSize; ++i) {
        ListNode *temp = lists[i];
        while (temp) {
            tempV.push_back(temp->val);
            temp = temp->next;
        }
    }
    sort(tempV.begin(), tempV.end());

    for (int j = 0; j < tempV.size(); ++j) {
        ListNode *a = new ListNode(tempV[j]);
        tempNode->next = a;
        tempNode = tempNode->next;
    }
    return ret->next;
}


TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    //出现在左子树、右子树，或者根节点
    //出现在根节点
    if (root == NULL || root == p || root == q) return root;
    //遍历左子树
    TreeNode *right = lowestCommonAncestor(root->right, p, q);
    TreeNode *left = lowestCommonAncestor(root->left, p, q);

    //左右都不为空，在root
    if (right && left)return root;
    //遍历右左子树
    return left == NULL ? right : left;
}
//队列
vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
    //先根节点进入，然后队列左右子节点顺序遍历。偶数行数翻转一下

    vector<vector<int>> ret;
    if (!root) return ret;
    queue<TreeNode *> p;
    p.push(root);
    int lev = 1;
    int levCount = 1;
    int levCountTemp = 0;
    while (!p.empty()) {
        vector<int> temp;
        //每一层的节点个数
        levCountTemp = 0;
        //子节点
        while (levCount > 0) {
            temp.push_back(p.front()->val);
            //左右顺序
            if (p.front()->left) {
                p.push(p.front()->left);
                levCountTemp++;
            }
            if (p.front()->right) {
                p.push(p.front()->right);
                levCountTemp++;
            }
            p.pop();
            levCount--;
        }
        //偶数
        if (lev % 2 == 0) reverse(temp.begin(), temp.end());//偶数左右顺序翻转;
        levCount = levCountTemp;
        ret.push_back(temp);
        lev++;

    }
    return ret;
}