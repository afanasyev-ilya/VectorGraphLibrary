from urllib.request import urlopen
from bs4 import BeautifulSoup
import pickle


def extract_number(line):
    if (line != None):
        digits_list = [int(s) for s in line if s.isdigit()]
        return int(''.join(map(str, digits_list)))
    return None


def find_info_on_page(text, pattern):
    for line in text.splitlines():
        if pattern in line:
            return line
    return None


def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def extract_category(line):
    if line != None:
        return remove_prefix(line, "Category")
    return None


def get_categories_links(url):

    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    links = []
    lines = soup.find_all('a')

    first_category = False

    for line in lines:
        if "Affiliation" in line:
            first_category = True
        if first_category == True:
            category_link = get_graph_url(str(line))
            links.append(url + "/" + category_link)
        if "Trophic" in line:
            break

    return links


def get_graph_url(line):

    line = str(line)

    url = ""
    url_start = False

    for s in line:           
        if s == '"' and url_start == True:
            break
        if url_start == True:
            url += s
        if s == '"' and url_start == False:
            url_start = True

    return url


def get_graph_name(line):
    name = ""

    name_start = False

    for s in line:
        if s == '<' and name_start == True:
            break
        if name_start == True:
            name += s
        if s == '>' and name_start == False:
            name_start = True

    return name


def get_graph_url_and_name(category_url):
    try:
        html = urlopen(category_url).read()
    except:
        print("Couldn't access " + category_url + " in get_graph_url_and_name() function")
        return None

    soup = BeautifulSoup(html, features="html.parser")
    ans = []
    lines = soup.find_all('a')
    for line in lines:
        url = get_graph_url(str(line))
        name = get_graph_name(str(line))
        if url.startswith("../../") and url != "../../":
            ans.append({"url": category_url + url, "name": name})
    return ans


def get_tsv_link(soup):
    lines = soup.find_all('a')
    for line in lines:
        if "Data as TSV" in line:
            return get_graph_url(line)
    return None


def graph_process(graph_url):
    try:
        html = urlopen(graph_url).read()
    except:
        print("Couldn't access " + graph_url + " in graph_process() function")
        return None

    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # drop blank lines
    page_text = '\n'.join(chunk for chunk in chunks if chunk)

    size = None
    try:
        size = extract_number(find_info_on_page(page_text, "Size"))
    except:
        print("Error with " + graph_url + " while getting size")    
    volume = None
    try:
        volume = extract_number(find_info_on_page(page_text, "Volume"))
    except:
        print("Error with " + graph_url + " while getting volume")
    category = None
    try:
        category = extract_category(find_info_on_page(page_text, "Category"))
    except:
        print("Error with " + graph_url + " while getting category")
    max_degree = None
    try:
        max_degree = extract_number(find_info_on_page(page_text, "Maximum degree"))
    except:
        print("Error with " + graph_url + " while getting max_degree")
    avg_degree = None
    try:    
        avg_degree = extract_number(find_info_on_page(page_text, "Average degree"))
    except:
        print("Error with " + graph_url + " while getting avg_degree")
    lcc_size = None
    try:
        lcc_size = extract_number(find_info_on_page(page_text, "Size of LCC"))
    except:
        print("Error with " + graph_url + " while getting size of LCC")
    diameter = None
    try:
        diameter = extract_number(find_info_on_page(page_text, "Diameter"))
    except:
        print("Error with " + graph_url + " while getting diameter")
    loop_cnt = None
    try:
        loop_cnt = extract_number(find_info_on_page(page_text, "Loop count"))
    except:
        print("Error with " + graph_url + " while getting loop_cnt")
    try:
        TSV = get_tsv_link(soup)
    except:
        print("Error with " + graph_url + " while getting diameter")
    tsv_url = graph_url
    if TSV != None:
        tsv_url += TSV
    else:
        tsv_url = None
    return {"size": size, "volume": volume, "category": category, "page_link": graph_url, "tsv_link": tsv_url, "max_degree": max_degree, "avg_degree": avg_degree, "lcc_size": lcc_size, "diameter": diameter, "loop_cnt": loop_cnt}


def get_graph_info(url):
    try:
        html = urlopen(url).read()
    except:
        print("Couldn't access " + url + " in get_graph_info() function")
        return None
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    categories_links = get_categories_links(url)

    ans = {}

    for category_url in categories_links:
        graphs = get_graph_url_and_name(category_url)
        if graphs is not None:
            for graph in graphs:
                ans[graph["name"]] = graph_process(graph["url"])

    return ans


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ans = get_graph_info("http://konect.cc/categories")
    save_obj(ans, "graphs")

