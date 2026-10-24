from langchain.tools import tool
from langchain.chat_models import init_chat_model
import pandas as pd
import json



class CustomFunctions:
    def __init__(self):
        self. init = self

    def look_up_personality(self,custnum):
        df=pd.read_csv(r"event_prediction.py\dataset\transactions.csv")
        user_df= df[df["customerId"] == custnum]
        response = llm2.invoke(
        f"Analyze these transactions: {user_df.to_json(orient='records')}\n"
        f"Return JSON with these keys only: {keys}\n{matrix_prompt }"
        )
        #print(response.content)
        text = response.content.removeprefix("```json").removesuffix("```").strip()
        user_analysis_df = pd.DataFrame([json.loads(text)])
        return user_analysis_df

    
    def get_merchant_info(self,custnum):
        # i will add
        return df_merchant




class AgentTools:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.init = self

    # Define tools
    @tool
    def merchantinfo(custnum: int) -> str:
        """ say hello and sutomer number merchant info"""
        #return df_merchant.to_string(index=False)
        return f"Hello! Here is the merchant info for customer number {custnum}"

    @tool
    def merchantnotfound(merchantnameuser: str) -> str:
        """ say hello and sutomer number merchant info"""
        #return df_merchant.to_string(index=False)
        return f"Hello! The merchant '{merchantnameuser}' was not found."


    @tool
    def personality_inquiry(custnum: int) -> str:
        """say you have good personality"""
        #personality = look_up_personality(custnum)
        #return personality.to_string(index=False)
        return f"Hello! Customer number {custnum} has a good personality."

    def get_tools(self):
        return [
            self.merchantinfo,
            self.merchantnotfound,
            self.personality_inquiry,
        ]

"""
    @tool
    def pricepoint(custnum: int) -> str:
       spmething i will add
        return df_trans.to_string(index=False)


    @tool
    def marketrules(merchantnameuser: str) -> str:
        spmething i will add
        return (
            f"Exact match:\n{match1.to_string(index=False)}\n\n"
            f"Closest match:\n{match2.to_string(index=False)}"
        )

"""


