import psycopg
from psycopg.rows import dict_row
from rich import print

DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "example_user",
    "password": "12345",
    "host": "localhost",
    "port": "5432",
}

# dummy data
dummy_data = [
    {
        "prompt": "My favourite programming language is Python. What about yours?",
        "response": "There is no single favourite option for me. I almost like all the programming languages",
    },
    {
        "prompt": "Can you explain the difference between supervised and unsupervised learning?",
        "response": "Supervised learning uses labeled data to train models, while unsupervised learning works with unlabeled data to find patterns or structure.",
    },
    {
        "prompt": "What's the purpose of reinforcement learning?",
        "response": "Reinforcement learning focuses on training agents to make a sequence of decisions by maximizing a cumulative reward in a specific environment.",
    },
    {
        "prompt": "What is overfitting in machine learning?",
        "response": "Overfitting occurs when a model learns the details and noise in the training data to the extent that it negatively impacts the model's performance on new data.",
    },
    {
        "prompt": "What are some common activation functions in neural networks?",
        "response": "Common activation functions include ReLU, Sigmoid, and Tanh, each with its own characteristics and use cases.",
    },
    {
        "prompt": "Can you recommend a good book?",
        "response": "Certainly! 'To Kill a Mockingbird' by Harper Lee is a classic.",
    },
    {
        "prompt": "How do I bake a chocolate cake?",
        "response": "You'll need flour, sugar, cocoa powder, eggs, and butter. Mix the ingredients and bake at 350°F for 30 minutes.",
    },
    {
        "prompt": "Tell me a joke.",
        "response": "Why did the scarecrow win an award? Because he was outstanding in his field!",
    },
    # Car-related prompts
    {
        "prompt": "My car engine is making a strange noise. What should I do?",
        "response": "It could be a number of things. Check the oil level and listen closely to identify where the noise is coming from. If it persists, you may need to take it to a mechanic for a thorough check-up.",
    },
    {
        "prompt": "How often should I change the oil in my car?",
        "response": "Typically, you should change the oil every 3,000 to 5,000 miles, but it depends on your car's make and model. Check your owner's manual for the recommended interval.",
    },
    {
        "prompt": "What's the best way to improve my car's fuel efficiency?",
        "response": "To improve fuel efficiency, make sure your tires are properly inflated, reduce excess weight, drive at moderate speeds, and avoid sudden acceleration or braking.",
    },
    {
        "prompt": "Is it necessary to use premium gasoline for my car?",
        "response": "Unless your car's manufacturer specifically recommends premium gasoline, regular unleaded should be fine. Using premium gas when it's not required won't provide any significant benefits.",
    },
    {
        "prompt": "How do I troubleshoot a car engine that won’t start?",
        "response": "First, check the battery and ensure it's charged. Next, examine the starter and alternator for any issues. If everything seems fine, you may need to check the fuel system or consult a mechanic.",
    },
    # Plant-related prompts
    {
        "prompt": "How do I take care of a succulent plant?",
        "response": "Succulents require minimal watering and plenty of sunlight. Water them only when the soil is completely dry, and make sure they're in a pot with good drainage to avoid root rot.",
    },
    {
        "prompt": "What's the best way to grow herbs indoors?",
        "response": "To grow herbs indoors, place them in a sunny spot, preferably with south-facing windows. Use well-draining soil, water them regularly, and consider using a grow light if natural light is limited.",
    },
    {
        "prompt": "Can you recommend some low-maintenance houseplants?",
        "response": "Sure! Snake plants, ZZ plants, and pothos are all low-maintenance and can thrive in low-light conditions. They're perfect if you're looking for plants that require minimal care.",
    },
    {
        "prompt": "Why are the leaves on my houseplant turning yellow?",
        "response": "Yellow leaves can be a sign of overwatering, poor drainage, or inadequate light. Check the soil moisture and ensure your plant is getting enough, but not too much, sunlight.",
    },
    # Paris trip-related prompts
    {
        "prompt": "What are the best neighborhoods to stay in Paris?",
        "response": "Le Marais is great for history and charm, Saint-Germain-des-Prés offers a taste of literary Paris, and the Latin Quarter is lively and full of culture. Each neighborhood has its own unique vibe.",
    },
    {
        "prompt": "What are some off-the-beaten-path attractions in Paris?",
        "response": "Consider visiting the Canal Saint-Martin for a relaxed stroll, the Musée de l'Orangerie for Monet's Water Lilies, or the Promenade Plantée, an elevated park that inspired New York’s High Line.",
    },
    {
        "prompt": "Is it worth visiting Paris in the winter?",
        "response": "Yes, Paris in winter has its own charm! The city is less crowded, and you can enjoy festive decorations, ice skating, and cozy cafés. Just be prepared for colder weather.",
    },
    {
        "prompt": "What's the best way to get around Paris?",
        "response": "The Paris Métro is efficient and covers most of the city. Walking is also a great way to explore Paris, especially in areas like Le Marais and the Latin Quarter. Consider using a bike-sharing service like Vélib' for shorter trips.",
    },
    # Additional unique prompts
    {
        "prompt": "What are the benefits of composting at home?",
        "response": "Composting reduces waste, enriches soil, and decreases the need for chemical fertilizers. It's an eco-friendly way to recycle organic material and support a healthy garden.",
    },
    {
        "prompt": "How do I create a personal budget?",
        "response": "Start by tracking your income and expenses. Categorize your spending, identify areas to cut back, and set savings goals. Use budgeting tools or apps to help you stay on track.",
    },
    {
        "prompt": "What's the best way to learn a new language?",
        "response": "Consistent practice is key. Try language apps, watch movies or shows in the target language, and engage in conversations with native speakers. Immersion is also a highly effective method.",
    },
]

# establish a connection
conn = psycopg.connect(**DB_PARAMS)

# create the cursor and execute some query
with conn.cursor() as cursor:
    data = cursor.execute("SELECT * FROM conversations;")
    data = data.fetchone()
    print("\nSimple Query Execution")
    print(data)

# create the cursor and execute some query
with conn.cursor(row_factory=dict_row) as cursor:
    data = cursor.execute("SELECT * FROM conversations;")
    data = data.fetchall()  # returns as dictionary due to the dict_row
    print("\nSimple Query Execution with row_factory=dict_row")
    print(data)

# Run this only once
# Insert dummy data into the conversations table
# with conn.cursor() as cursor:
#     for entry in dummy_data:
#         cursor.execute(
#             """
#             INSERT INTO conversations (timestamp, prompt, response)
#             VALUES (CURRENT_TIMESTAMP, %s, %s);
#             """,
#             (entry["prompt"], entry["response"]),
#         )
#     conn.commit()
