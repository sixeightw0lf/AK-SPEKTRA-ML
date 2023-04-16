import os
import sys
import requests
import html2text
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_html(url, use_js=False, proxy=None, user_agent=None):
    if use_js:
        options = Options()
        if proxy:
            options.add_argument('--proxy-server=%s' % proxy)
        if user_agent:
            options.add_argument('user-agent=%s' % user_agent)
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        html = driver.page_source
        driver.quit()
    else:
        headers = {'User-Agent': user_agent} if user_agent else None
        response = requests.get(url, headers=headers, proxies={'http': proxy, 'https': proxy} if proxy else None)
        html = response.content

    return html

def parse_html(html, selectors=None):
    soup = BeautifulSoup(html, 'html.parser')
    if selectors:
        elements = []
        for selector in selectors:
            elements += soup.select(selector)
        html = ''.join(str(element) for element in elements)
    return html

def convert_to_markdown(html):
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    markdown = h2t.handle(html)
    return markdown

def save_markdown(markdown, filename):
    output_dir = 'app/markdown/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(markdown)

def main():
    input_type = input("Enter input type (url, html, text): ")
    input_value = input("Enter the input (URL, HTML file path, or text file path): ")
    output_filename = input("Enter output filename: ")

    selectors = input("Enter element selectors separated by commas (optional): ")
    selectors = [s.strip() for s in selectors.split(',')] if selectors else None

    use_js = input("Is the website rendered with JavaScript? (y/n, default=n): ")
    use_js = True if use_js.lower() == 'y' else False

    proxy = input("Enter proxy (optional): ")
    user_agent = input("Enter user agent (optional): ")

    if input_type == 'url':
        html = get_html(input_value, use_js=use_js, proxy=proxy, user_agent=user_agent)
    elif input_type == 'html':
        with open(input_value, 'r') as f:
            html = f.read()
    elif input_type == 'text':
        with open(input_value, 'r') as f:
            html = f.read()
    else:
        print("Invalid input type.")
        sys.exit(1)

    parsed_html = parse_html(html, selectors=selectors)
    markdown = convert_to_markdown(parsed_html)
    save_markdown(markdown, output_filename)
    print(f"Markdown file saved to app/markdown/{output_filename}")

if __name__ == "__main__":
    main()
