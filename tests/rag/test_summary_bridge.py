import asyncio

from coursepilot.rag.summary_bridge import SummaryBridge



async def main():
    units = [
        {"content": r"""11.2.2比值判别法和根值判别法
用比较判别法判定正项级数的敛散性，依赖于另一个已知敛散性的适当正项级数.但有时候，要选择这样一个级数并不容易.为此我们介绍仅依赖级数本身结构来确定其敛散性的两个常用判别法—比值判别法和根值判别法，其中比值判别法也称为d’AIembert（1717—1783，法国数学家）判别法，根值判别法也称为Cauchy判别法.
238
第11章级 数
定理11.8（比值判别法）设 $\sum _ { n = 1 } ^ { \infty } a _ { n }$ 为正项级数，且 $\operatorname* { l i m } _ { n \to \infty } { \frac { a _ { n + 1 } } { a _ { n } } } = l$ （或者 $+ \infty )$ ,那么
（1）当 $0 \leqslant l < 1$ 时，级数 $\sum _ { n = 1 } ^ { \infty } a _ { n }$ 收敛；
(2）当 $1 < l \leqslant + \infty$ 时，级数 $\sum _ { n = 1 } ^ { \infty } a _ { n }$ 发散.
证（1）当 $0 \leqslant l < 1$ 时，取 $\varepsilon = \frac { 1 - l } { 2 } > 0$ 由 $\operatorname* { l i m } _ { n \to \infty } { \frac { a _ { n + 1 } } { a _ { n } } } = l$ 可得： $N _ { 1 } \in \mathbf { Z } ^ { + }$ ,当$n > N _ { 1 }$ 时，
$$
\left| \frac { a _ { n + 1 } } { a _ { n } } - l \right| < \varepsilon = \frac { 1 - l } { 2 } \Rightarrow 0 \leqslant \frac { a _ { n + 1 } } { a _ { n } } < l + \frac { 1 - l } { 2 } = \frac { 1 + l } { 2 } ,
$$
因此当 $n > N _ { 1 }$ 时有
$$
a _ { n + 1 } < \frac { 1 + l } { 2 } a _ { n } < \cdots < \left( \frac { 1 + l } { 2 } \right) ^ { n - N _ { 1 } } a _ { N _ { 1 } } .
$$
由于 $\frac { l + 1 } { 2 } < 1$ ，因此等比级数""", "summary": ""}
    ]

    bridge = SummaryBridge()

    result = await bridge.run(units)
    print(result[0].get("summary"))


asyncio.run(main())
