# Customer Insights Agent

A simple command-line agent for exploring customer transaction data. It answers questions about a customer's merchants, visits, subscriptions, points, and shopping patterns.

## Setup

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project folder with your Google AI key:

   ```env
   GEMINI_API_KEY=ADD_YOUR_GOOGLE_KEY_HERE
   ```

   Replace the placeholder with your key. Keep `.env` private; do not commit it.

3. Make sure `transactions.csv` is in the location expected by the code.

4. Start the agent:

   ```bash
   python agent.py
   ```

5. When prompted, ask about a customer using their customer number and, when relevant, a merchant name.

## Example questions

- How many points did this customer earn?
- What are the minimum and maximum transaction amounts?
- List the customer's subscriptions.
- What personality signals can you infer from this customer's transactions?
- List the merchants this customer has shopped at.
- If this customer shopped at Merchant X, what other merchants might they shop at?

Use the terms and fields available in `transactions.csv`; the agent can only answer from the data and analysis implemented in the project.

## How it works

The project uses LangChain to connect a Gemini chat model to tools written for this project. The agent reads a question, chooses a relevant tool when needed, and uses the tool's result to form a response. The tools perform the actual lookups and calculations on the transaction data.

Personality signals are patterns suggested by shopping behavior, such as frequently visited merchant categories, repeat visits, and subscription habits. These are clues for understanding preferences, not definitive claims about a person.

For merchant recommendations, the project can use market basket analysis. Apriori is a basic algorithm that finds items (here, merchants) that often appear together in transactions. Those co-occurrence patterns can suggest other merchants a customer may be interested in; they are associations, not guarantees.
