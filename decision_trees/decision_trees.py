#!/usr/bin/env python
# coding: utf-8

# ### Decision Trees

# Task 1

# In[450]:


class Tree:
    '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

      Examples:
        l1 = Tree.leaf("leaf1")
        l2 = Tree.leaf("leaf2")
        tree = Tree(data="root", left=l1, right=Tree(right=l2))
    '''

    def leaf(data):
        '''Create a leaf tree
        '''
        return Tree(data=data)

  # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right)

  # all arguments after `*` are *keyword-only*!
    def __init__(self, *, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        '''Check if this tree is a leaf tree
        '''
        return self.left == None and self.right == None

    def children(self):
        '''List of child subtrees
        '''
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        '''Compute the depth of a tree
        A leaf is depth-1, and a child is one deeper than the parent.
        '''
        return max([x.depth() for x in self.children()], default=0) + 1


# In[445]:


tree = Tree(data='isSystems?', left='like', 
            right=Tree(data='TakenOtherSys?', left=Tree(data='morning?', left='like', right='nah'), 
                       right=Tree(data='likedOtherSys?', left='nah', right='like')))


# In[98]:


print(tree)


# Task 2

# In[109]:


import pandas as pd
from io import StringIO


# In[110]:


csv_string = '''rating,easy,ai,systems,theory,morning
 2,True,True,False,True,False
 2,True,True,False,True,False
 2,False,True,False,False,False
 2,False,False,False,True,False
 2,False,True,True,False,True
 1,True,True,False,False,False
 1,True,True,False,True,False
 1,False,True,False,True,False
 0,False,False,False,False,True
 0,True,False,False,True,True
 0,False,True,False,True,False
 0,True,True,True,True,True
-1,True,True,True,False,True
-1,False,False,True,True,False
-1,False,False,True,False,True
-1,True,False,True,False,True
-2,False,False,True,True,False
-2,False,True,True,False,True
-2,True,False,True,False,False
-2,True,False,True,False,True'''


# In[111]:


csv_data = StringIO(csv_string)


# In[112]:


df = pd.read_csv(csv_data)


# In[114]:


df['ok'] = df['rating'] >= 0


# In[150]:


print(df)


# Task 3

# In[121]:


import numpy as np


# In[122]:


def single_feature_score(data, goal, feature):
    res = data[feature] == data[goal]
    score = np.sum(res) / len(data) * 100
    if score < 50:
        score = 100 - score
    return score


# In[194]:


features = df.columns[1:-1]


# In[196]:


for feature in features:
    score = single_feature_score(df, 'ok', feature)
    print(f'{feature} score: {score}')


# In[324]:


def get_best_feature(data, goal, features):
    # optional: avoid the lambda using `functools.partial`
    return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[325]:


print(get_best_feature(df, 'ok', features))


# In[326]:


def get_worst_feature(data, goal, features):
    # optional: avoid the lambda using `functools.partial`
    return min(features, key=lambda f: single_feature_score(data, goal, f))


# In[327]:


print(get_worst_feature(df, 'ok', features))


# Task 4

# In[213]:


from collections import Counter


# In[474]:


def DecisionTreeTrain(data, goal, remaining_features) -> Tree:

    guess = max(Counter(data[goal].values))

    labels = []
    for feature in remaining_features:
        labels.extend(data[feature].values)

    if len(set(labels)) == 1 or not remaining_features:
        return Tree.leaf(guess)

    best_feature = get_best_feature(data, goal, remaining_features)
    remaining_features.remove(best_feature)

    left = DecisionTreeTrain(
        df[df[best_feature] == False], goal, remaining_features)
    right = DecisionTreeTrain(
        df[df[best_feature] == True], goal, remaining_features)

    return Tree(data=best_feature, left=left, right=right)


# In[476]:


tree = DecisionTreeTrain(data, 'ok', list(data.columns[:-1]))


# In[484]:


print(tree)


# Task 5

# In[ ]:


def DecisionTreeTrain(data, goal, remaining_features, max_depth=None) -> Tree:

    guess = max(Counter(data[goal].values))

    labels = []
    for feature in remaining_features:
        labels.extend(data[feature].values)

    if max_depth == 1 or len(set(labels)) == 1 or not remaining_features:
        return Tree.leaf(guess)

    best_feature = get_best_feature(data, goal, remaining_features)
    remaining_features.remove(best_feature)
    
    if max_depth:
        max_depth -= 1

    left = DecisionTreeTrain(
        df[df[best_feature] == False], goal, remaining_features, max_depth=max_depth)
    right = DecisionTreeTrain(
        df[df[best_feature] == True], goal, remaining_features, max_depth=max_depth)

    return Tree(data=best_feature, left=left, right=right)
    