from duckduckgo_search import DDGS


def search(query):
    ddgs = DDGS()
    ddgs.text(query)
