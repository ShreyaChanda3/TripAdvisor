# TripAdvisor Web Crawling and Analysis Project

## Overview

This project involves web crawling, data extraction, and comprehensive analysis of user reviews from the TripAdvisor website. We utilized various techniques to perform descriptive analysis, visualization, regression analysis, and sentiment analysis on the collected data.

## Project Objectives

- **Extract comprehensive review data**: Including reviewer names, ratings, contributions, locations, and review dates.
- **Store data systematically**: In a SQLite database for ease of access and analysis.
- **Perform detailed analysis**: Including descriptive statistics, visualizations, regression analysis, and sentiment analysis.
- **Gain insights from user reviews**: To understand trends, patterns, and sentiments in customer feedback.

## Data Crawling and Storage

### Crawled Data

The data extracted from TripAdvisor reviews includes:

- Reviewer Name
- Rating
- Contribution
- Reviewer Location
- Written Date
- Written Year

### Crawling Process

1. **Identifying Elements**: We identified the HTML elements and tags associated with the required data.
2. **Tool Utilization**: Tools like Regex101 were used for real-time debugging of extraction expressions.

### Database Storage

The extracted data was stored in a SQLite database to facilitate further analysis.

## Analysis

### Descriptive Analysis

Conducted to understand the distribution of ratings and user contributions.

### Visualization

Various visualizations were created to illustrate the data, including:

- **Bubble Charts**: Showing the relationship between the first 20 customers and their contributions.
- **Word Clouds**: Displaying the most frequently used words in reviews.

### Regression Analysis

Used to identify relationships between different variables within the dataset.

### Sentiment Analysis

Performed to determine the underlying sentiment in the review text. Most sentiment polarity values ranged between 0 to 1, indicating predominantly positive reviews.

## Conclusion

### Technical Achievements

- **Efficient Web Crawling**: Successfully implemented an efficient web crawling process.
- **Enhanced Python Skills**: Gained significant confidence in Python coding.
- **Comprehensive Analysis**: Performed descriptive, regression, and sentiment analyses, deriving valuable insights from the data.
- **Insightful Visualizations**: Generated visualizations that provided a clear understanding of the data trends.
- **Effective Text and Sentiment Analysis**: Utilized text and sentiment analysis to identify data trends and outliers with ease.

## References

- [Web Crawling vs Web Scraping](https://brightdata.com/blog/leadership/web-crawling-vs-web-scraping)
- [Python Web Scraping Guide](https://brightdata.com/blog/how-tos/web-scraping-with-python)
- [Basic Bubble Plot with Matplotlib](https://www.python-graph-gallery.com/270-basic-bubble-plot)
- [Sentiment Analysis with Python](https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6)
- [Building Word Cloud in Python](https://www.analyticsvidhya.com/blog/2021/05/how-to-build-word-cloud-in-python/)
