class DataDTO:
    def __init__(
        self,
        user_prompt: str,
        proposer_count: int,
        aggregator_count: int,
        rating_agent: bool,
        selector_agent: bool,
        use_mct: bool,
        max_children: int,
        iteration: int,
        temp: float = 0.8,
        top_p: float = 0.8,
    ):
        self.user_prompt = user_prompt
        self.temp = temp
        self.top_p = top_p
        self.proposer_count = proposer_count
        self.aggregator_count = aggregator_count
        self.rating_agent = rating_agent
        self.selector_agent = selector_agent
        self.use_mct = use_mct
        self.max_children = max_children
        self.iteration = iteration