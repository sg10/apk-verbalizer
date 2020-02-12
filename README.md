# Code between the Lines: Semantic Analysis of Android Applications

This is the implementation of our framework to generate natural language text describing the purpose and core functionality of Android applications based on their actual code, as presented at IFIP SEC 2020.
See the [paper](https://graz.pure.elsevier.com/en/publications/code-between-the-lines-semantic-analysis-of-android-applications) by Feichtner and Gruber for more details.

In this repository you find our solution for:

- Deriving concise keywords and short phrases that indicate the main purpose of apps,
- Extracting semantic knowledge from resource identifiers, string constants, and API calls,
- Explaining predictions using the SHAP algorithm.

## Abstract

> Static and dynamic program analysis are the key concepts researchers apply to uncover security-critical implementation weaknesses in Android applications. As it is often not obvious in which context problematic statements occur, it is challenging to assess their practical impact. While some flaws may turn out to be bad practice but not undermine the overall security level, others could have a serious impact. Distinguishing them requires knowledge of the designated app purpose.
>
> In this paper, we introduce a machine learning-based system that is capable of generating natural language text describing the purpose and core functionality of Android apps based on their actual code. We design a dense neural network that captures the semantic relationships of resource identifiers, string constants, and API calls contained in apps to derive a high-level picture of implemented program behavior. For arbitrary applications, our system can predict precise, human-readable keywords and short phrases that indicate the main use-cases apps are designed for.
>
> We evaluate our solution on 67,040 real-world apps and find that with a precision between 69% and 84% we can identify keywords that also occur in the developer-provided description in Google Play. To avoid incomprehensible black box predictions, we apply a model explaining algorithm and demonstrate that our technique can substantially augment inspections of Android apps by contributing contextual information.

## Setup

```bash
conda install -c anaconda keras nltk pillow scikit-learn
conda install -c conda-forge langdetect shap
pip install androguard javalang
```

Sample output the framework produces can be found at https://sg10.github.io/apk-verbalizer/