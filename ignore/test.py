# Example: Scrape truthsocial.com/@realDonaldTrump using Playwright
from playwright.sync_api import sync_playwright

def get_latest_truth_post():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://truthsocial.com/@realDonaldTrump")
        page.wait_for_timeout(5000)  # Wait for JavaScript to load

        # Example: Grab first post text
        posts = page.query_selector_all('div[data-testid="post"]')
        latest = posts[0].inner_text() if posts else "No post found"
        browser.close()
        return latest
    
print(get_latest_truth_post())