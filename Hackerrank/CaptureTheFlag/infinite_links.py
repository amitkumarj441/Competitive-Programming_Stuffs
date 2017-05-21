import urllib2


def parse(html):
    key = None
    links = []

    start_token = "</b>"
    end_token = "</font>"

    start_pos = html.find(start_token)
    end_pos = html.find(end_token)
    if start_pos > -1 and end_pos > -1:
        key = html[start_pos + len(start_token):end_pos]

    start_token = "href="
    end_token=".html"
    old_pos = 0

    while old_pos >= 0:
        start_pos = html.find(start_token, old_pos, len(html))
        end_pos = html.find(end_token, start_pos, len(html))

        if start_pos < 0 or end_pos < 0:
            break

        link = html[start_pos + len(start_token): end_pos]
        links.append(link)
        old_pos = end_pos

    return key, links


def dfs():
    base_url = 'https://cdn.hackerrank.com/hackerrank/static/contests/capture-the-flag/infinite/'
    s = ['qds']
    keys = []
    visited = set(['qds'])

    while s:
        current = s.pop()
        print current

        response = urllib2.urlopen(base_url + current + ".html")

        secret, links = parse(response.read())

        if secret is not None:
            keys.append(secret)

        for link in links:
            if link not in visited:
                s.append(link)
                visited.add(link)

    return keys


def main():
    keys = dfs()

    for key in sorted(keys):
        print key


if __name__ == "__main__":
	main()
