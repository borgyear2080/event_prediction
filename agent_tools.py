from langchain.tools import tool
from langchain.chat_models import init_chat_model
import pandas as pd
import json
from llm import LLMs
import re


class CustomFunctions:
    def __init__(self):
        _, self.llm2 = LLMs().get()
        self.transactions = pd.read_csv(r"dataset\transactions.csv")
        self.rules = pd.read_csv(r"dataset\rules1.csv")

    def look_up_personality(self,custnum):
        df=pd.read_csv(r"dataset\transactions.csv")
        user_df= self.transactions[self.transactions["customerId"] == custnum]

        if user_df.empty:
            user_df = pd.DataFrame([json.loads("user is not found")])
            return user_df.to_string(index=False)
        
        how_to_score = """
        scale is  1 - 5 i is low 5 is high
        1. risk_tolerance:
        Format (1-5): Do analysis of data of transactions. Evidence of spending velocity despite low remaining availableMoney or high currentBalance relative to creditLimit.
        Output: Numeric value and transaction evidence, or null if spending stays well below thresholds. Scale

        2. responsible_spending:
        Format (1-5): Frequency and proportion of transactionAmount values remaining strictly within the observed available budget and balance. Any erratic spending
        in weekends laste night or high amount transactions. eg : high spend in pubs , gifts travels or others. 
        Output: Numeric value tracking budget alignment against limits, or null if data is insufficient.

        3. luxury:
        Format (1-5): Proportional share or explicit evidence of high-value transactionAmount records at luxury, high-end, or premium merchantName brands.
        or likes spend on cars, home , food, gifts etc.
        Output: Numeric value backed by identified premium merchants, or null if no premium merchants appear.

        4. budget:
        Format (1-5): Operational consistency of spending patterns appearing tightly controlled relative to availableMoney or creditLimit over time.
        is spending or got refunds , is eating in budget restraudants , and no big gifts etc.
        Output: Numeric consistency score with supporting volume evidence, or null if data is too sparse.

        5. subscriptions:
        Format (List): Identification of recurring, identical transactionAmount values at fixed intervals for subscription-based merchantName entries.
        Output: A structured list of recurring subscription merchants found, or an empty list if none are detected.

        6. happiness:
        Format (1–5): this can be a combination of health, social, Pure transactional evidence of voluntary expenditures on leisure, hobby, or entertainment merchantCategoryCode entries.
        Output: Numeric score of enjoyable activities without claiming emotional status, or null if insufficient.

        7. interests:
        Format (List): Frequent or recurring groupings of specific activity sectors like books, gaming, music, or hobbyist merchantCategoryCode markers.
        Output: List of recurring behavioral/merchant categories based strictly on frequency, or an empty list.

        8. travel:
        Format (1-5): Explicit evidence of travel-related purchases such as airlines, hotels, car rentals, or cross-border merchantCountryCode discrepancies.
        any change in merchant or zip codes
        Output: Numeric score indicating travel transaction presence and scale, or null if absent.

        9. social:
        Format (1-5): if person has social life. has visited social events and see bills if it is one person or multiple.
        Transactional evidence of shared venues, group entertainment, or social spaces without inferring personal relationships.
        Output: Numeric value mapping social merchant interactions, or null if no such venues are recorded.

        10. health:
            Format (1-5): 5 is gym visits. too much pub , drinking is 1. normal pub vistits is not negative.
            Tangible evidence of healthcare, medical, pharmacy, or wellness-related spending without inferring actual health status.
            Output: Numeric score summarizing volume of health-related transactions, or null if none exist.

        11. sports:
            Format (1-5): is interested in sports. Transactional footprint of sports equipment, gym memberships, fitness facilities, or active recreation merchantCategoryCode entries.
            Output: Numeric score indicating sports/fitness procurement history, or null if no records exist.

        12. food:
            Format (1-5): is interested in food, diners. Raw frequency and total volume allocation towards groceries, dining, and restaurants without inferring household size.
            Output: Numeric score reflecting relative food expenditure volume, or null if baseline data is missing.

        13. romance:
            Format (1-5): Is interested in romance if have visited spa, flowers, chocolates, and bill is for 2 when mostly it sis for 1.
            make up and interests in looks and such activities. Enjoys music , theater and such things.
            Output: null universally unless transactions explicitly isolate an undeniable romantic service purchase.

        14. self_care:
            Format (1-5): Measurable expenditure evidence at personal care merchants, including salons, spas, grooming, or personal hygiene services.
            include reading, health care, grooming and taking care of dental, eyeware and looks.
            Output: Numeric value summarizing personal care transaction frequency, or null if undocumented.

        15. lonely:
            Format (1-5): bills are mostly of 1 person. or has indications the increase in bills or outings of 1 person.
            is expenses include a lot of netflix and pubs. alone outings. or lack of social activities or reduced activities.
            Output: numeric value

        16. education:
            Format (0–5): Clear evidence of spending at universities, schools, tuition centers, textbook vendors, or educational merchantCategoryCode entities.
            interest in reading , subscriptions etc
            Output: Numeric value scaling educational investments, or null if no such records exist.

        17. pubs:
            Format (1-5): Volumetric frequency and monetary density of pub, bar, or nightlife vendor transactions without inferring personal alcohol consumption.
            you can check time and days of week these visit happen. 
            Output: Numeric score mapping bar/pub establishment interaction frequency, or null if zero transactions.

        18. change_in_behaviour:
            Format (String): Qualitative evaluation of longitudinal shifts in transaction metrics; requires multi-month data to confirm structural deviations.
            Output: Concise text description of a notable spending pattern shift, or exactly "no clear change". limit to 2 sentences

        19. what_changed:
            Format (String): Granular identification of precisely which categories, frequencies, or monetary values mutated over the observed tracking period.
            Output: Text isolating specific category/amount modifications across months, or null if timeline data is insufficient. limit to 2 sentences

        20. introvert:
            Format (string): if person has bills of 1 person , spending of 1 person, not a lot of social visits
            Output: 2 lines of text

        21. family:
            Format (1-5) if bills are of more than 1 person and more the 2. high grovery or home store spendings, restraudants that have bills or a family amount
            visits to holiday parks and moviews etc that are of > 2,3 people. any spends on children 
            Output:numeric 1-5
        22.summary_personality:
            from all the above anaysis i want you to tell me a personality summary of this person. inclue specifics and your reasons how you concluded above.
            summary cna include what all food, places, things person likes. can be 4 sentences of specific details.
            Add information of what subscriptionsthey have and merchants they buy from. What stores they buy from . If they dine alone or with friends. 
            what do they like , dislike, and habits. 
            Output: str 4 lines of text
        """
        
        keys = ["risk_tolerance", "responsible_spending", "luxury", "budget", "subscriptions", "happiness", "interests", "travel", "social", "health", "sports", "food", "romance", "self_care", "lonely", "education", "pubs", "change in behaviour", "what changed", "introvert", "family","summary_personality"]

        matrix_prompt = f"""
        You are a highly elite, unapologetically nosey Personal Behavioral Spy and financial detective. As the saying goes: "Follow the money, because money talks." You have been granted explicit, consensual access to a target's private spending history, and your job is to peer deep into their bank statements to dissect their secret lifestyle habits, vices, and true personality. Do not hold back—be detective mode, sharp, and brutally insightful. 
        Carefully analyze the provided spending data using these evaluation rules:
        {how_to_score}

        You must output your final espionage assessment strictly as a JSON object matching the exact keys provided below. Do not include any conversational preamble, markdown code blocks (like ```json), or trailing commentary. Output only the raw valid JSON.

        Required Keys:
        {keys}

        """
        
        response = self.llm2.invoke(
        f"Analyze these transactions: {user_df.to_json(orient='records')}\n"
        f"Return JSON with these keys only: {keys}\n{matrix_prompt }"
        )
        text = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        user_analysis_df = pd.DataFrame([json.loads(text)])
        return user_analysis_df.to_string(index=False)

    
    def rules_lookup(self,merchantnameuser):
        rules = self.rules
        matches_conf = rules[
            rules["antecedents"].apply(lambda items: merchantnameuser in str(items))
        ].sort_values("confidence", ascending=False)
        
        matches_lift = rules[
            rules["antecedents"].apply(lambda items: merchantnameuser in str(items))
        ].sort_values("lift", ascending=False)
        
        match1 =matches_conf[["consequents", "confidence", "lift"]].head(9)
        match2 =matches_lift[["consequents", "confidence", "lift"]].head(10)
        
        return match1,match2




class AgentTools:
    def __init__(self):
        #we call all the functions the tools will need for processing
        custom = CustomFunctions()

        @tool
        def merchantinfo(custnum: int) -> str:
            """look up df with customer num and send the df with merchant info need list of ALL merchant names . 
            list out all merchants. find out what they like and interests this will be used to recommend them other things
            so list every merchant , interests etc and which they visit more. any subscriptions habits etc 
            I just need top 10 merchants they visit and the frequences of visits. NOT ALL.
            """
            df=custom.transactions
            df_merchant= df.loc[df["customerId"] == custnum].merchantName
            return df_merchant.to_string(index=False)

        @tool
        def merchantnotfound(merchantnameuser: str) -> str:
            """Report that a merchant was not found."""
            return f"The merchant '{merchantnameuser}' was not found."

        @tool
        def personality_inquiry(custnum: int) -> str:
            """Get personality signals for a customer. Response must be a well structured list of all the signals numbered and the details of each signal.
            Add details and names of merchants the analysis of amounts and frequences of transactions and the details of the merchants.. Include your reasoning and how you concluded the signals. Include details of subscriptions and merchants they buy from. What stores they buy from. If they dine alone or with friends. what do they like , dislike, and habits.
            i want details.
            """
            result = custom.look_up_personality(custnum)
            return f"Customer Personality Signal {custnum}: {result}"

        @tool
        def pricepoint(custnum: int) -> str:
            """look up df with customer num and send the df with the range of the price points of thier transactions
            examin the subscriptions amounts, max spend amounts and average spend amounts and return back . 
            Subscriptions price, max spend amount, min and average. simple 3 lines. short answer prefered.
            Dont OVER ANALYZE. I dont need all transactions. just  min max average and subscriptions. i want to know the price points of this person. Also any repeat patterns of spendings.
            """
            df=custom.rules
            df_trans= df.loc[df["customerId"] == custnum][["transactionAmount",	"merchantName"]]
            if df_trans.empty:
                return f"No transactions found for customer."
            return df_trans.to_string(index=False)


        @tool
        def marketrules(merchantnameuser: str) -> str:
            """Find the closest merchant name in the dataframe, then return its top 10
            recommended merchants from the market-basket rules, ranked by lift/confidence.
            Get closest matches for confidence and lift scores match 1 is confidence and match2 is lift
            Return both matches as readable text. If there are no matches, say so.
            send me 2 lists of merchant names in readable format
            **if merchant name is not found then use tool merchantnotfound and then use tool marketrules
            """
            match1, match2 = custom.rules_lookup(merchantnameuser)
            return (
                f"Exact match:\n{match1.to_string(index=False)}\n\n"
                f"Closest match:\n{match2.to_string(index=False)}"
            )


        self.tools = [merchantinfo, merchantnotfound, personality_inquiry,marketrules, pricepoint]

    def get_tools(self):
        return self.tools
    




