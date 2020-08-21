關於「統計建模技法」
============================


「心理計量學」（psychometrics）為一探討如何使用數量模型，刻畫人類外顯行為與內在歷程之學門。典型的「心理計量模型」（psychometric model）包含「因素分析」（factor analysis）、「試題反應理論」（item response theory）等試圖刻畫潛在構念（latent constructs）與觀察指標（observed indicators）間關係之方法。因此，潛在構念（或稱潛在變項）的出現，可說是「心理計量模型」的一大特色，而如何處理潛在變項對於統計推論與心理學研究帶來的挑戰，則為「心理計量學」的核心議題。

「心理計量學」可視為「統計學」與「心理學」兩學門領域的結合，因此，在進行「心理計量學」研究時，除了心理學的知識外，也需要具備以下三類的技能：

1. 數理技能（mathematical skills），包括線性代數（linear algebra）、微積分（calculus）、以及機率（probability）。

2. 統計理論（statistical theory），包括建立估計（estimation）與推論（inference）程序的技術。

3. 程式設計（programming），包括如何在電腦上進行線性代數之計算、數值微分與積分、以及機率物件的操弄。

從個人的角度，「統計學」可分為「建模（modeling）」與「推論（inference）」兩個議題。「建模」指的是如何使用數量模型來合理地刻畫現象（如使用線性或是二次式來描述兩變項間的關係），而「推論」指的是如何考慮到資料的隨機性以對模型參數進行合理之解釋（如對迴歸係數進行估計與檢定）。儘管在理論統計（theoretical statistics）的領域，仍不斷地有新的「推論」方法被發展出來，但在統計的實務應用時，研究者大多採用相對老舊的「推論」方法，但透過不同的「建模」策略來對資料背後的科學現象以更深入的了解。因此，「建模」可說是當代量化研究中很重要的一個面向。

此頁面放置了黃柏僩老師所撰寫的「統計建模技法」（Techniques in Statistical Modeling，TISM）講義，其旨在介紹「統計建模」所需的基本技術，以及如何使用概似函數（likelihood function）對模型進行評估。此外，本講義也將以「心理計量學」常使用之模型作為範例，並搭配python的深度學習套件PyTorch進行實作。

此講義主要用於訓練量化方法主修研究生之課程，講義本身並無法達到「自給自足」（self-contained）的目標，因此，對於交代不清的部分，我盡可能地會附上參考文獻或是外部連結，讀者可以自行將該缺失的部分補足。

在閱讀此講義之前，讀者需具有以下之先備知識：

1. 數理基礎，包括向量（vector）與矩陣（matrix）之運算規則、微分（differentiation）與積分（integral）之幾何意義、機率分配與期望值（expectation）。

2. 統計學的基礎，包括平均數、變異數、相關係數之計算，估計、假設檢定（hypothesis testing）、與信賴區間（confidence interval）之概念。

3. python程式設計基礎，包括基本資料類型、流程控制（flow control）、函數等概念。若無基礎，Jake VanderPlask 的 [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython) 會是一很好的學習資源。

本講義之內容，皆預設讀者已具備前述之基礎。讀者不需精熟這些先備知識，但應至少須知道這些概念的定義，以及不害怕深入了解這些概念。

若您發現講義的內容有誤，請將勘誤寄至本人之電子信箱 psyphh@gmail.com，感謝！
