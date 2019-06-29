//
// Created by william on 19-6-2.
//

#ifndef ALG_MINSTACH_H
#define ALG_MINSTACH_H

#include <stack>

class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {

    }

    void push(int x) {
        if (min.empty() || x <= min.top()) {
            min.push(x);
        }
        s.push(x);
    }

    void pop() {
        if (s.top() == min.top()) {
            min.pop();
        }
        s.pop();
    }

    int top() {
        return s.top();
    }

    int getMin() {
        return min.top();
    }

private:
    std::stack<int> min;
    std::stack<int> s;

};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
#endif //ALG_MINSTACH_H
