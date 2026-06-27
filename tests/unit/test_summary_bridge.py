import asyncio

from coursepilot.rag.summary_bridge import SummaryBridge



async def main():
    units = [
        {"content": r"""$4 . 8 3 9 \times 1 0 ^ { 1 8 } g .$
18. ${ \sqrt { \frac { 2 } { 3 } } } R .$
19. $\left( 0 , 0 , \frac { h } { 4 } \right) , \frac { \pi a ^ { 4 } h } { 1 0 } .$""", "summary": ""}
    ]

    bridge = SummaryBridge()

    result = await bridge.run(units)
    print(result[0].get("summary"))


asyncio.run(main())
