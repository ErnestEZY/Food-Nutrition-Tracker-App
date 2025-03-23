# Food-Nutrition-Tracker-App
*Overview*

The Food Nutrition Tracker Pro WebApp is a user-friendly application designed to help you monitor and analyze your daily food intake, track progress toward your dietary goals, and gain insights into your nutrition habits. Built with Streamlit, this app provides an interactive interface to log food, visualize nutrient trends, and receive personalized recommendations based on your BMI and diet preferences.

*Features*
1. Daily Food Logging: Log your meals and track calories, protein, carbohydrates, and fats.
2. Daily Progress Tracking: View your progress toward daily nutrient goals with visual progress bars.
3. Macronutrient Distribution: See a pie chart of your daily macronutrient breakdown.
4. Goal Achievement Trends: Analyze your progress over the past 7 days with a bar chart showing the percentage of goals achieved for each nutrient.
5. 7-Day Goal Streak: Track your consistency in meeting daily goals with a weekly streak table.
6. Nutrient Trends: Visualize your nutrient intake over the past 7 days with a line chart.
7. Personal Insights: Get a daily nutrition score and a predicted calorie intake for the next day.
8. Top Foods Consumed: View a treemap of your top calorie-contributing foods over the past 7 days.
9. BMI-Based Recommendations: Receive dietary suggestions based on your BMI.

*Prerequisites*

To run the Nutrition Tracker WebApp, you’ll need the following:
-Python 3.8 or higher
-A MongoDB database (for storing food logs)
-Streamlit and other required Python libraries (listed in requirements.txt)

*Usage*
1. Calculate Your BMI: Start on the Home page by entering your height and weight to calculate your BMI.
2. Select a Diet Type: Choose a diet type (e.g., Standard, Keto) to set personalized nutrient goals.
3. Log Your Food: Go to the "Daily Food Log" page to add your meals and their nutritional information.
4. View Your Analysis: Navigate to the "Nutrition Analysis" page to see your daily progress, trends, and insights.
5. Check your daily nutrient intake and progress toward goals.
6. Explore historical trends and streaks.
7. Review personalized insights and recommendations.

*Notes*

•The app uses Malaysia Standard Time (MST) for date and time calculations.
•Daily totals reset at 12 AM each day, but historical data is retained for trend analysis.
•Ensure your MongoDB database is properly configured to store and retrieve food logs.
