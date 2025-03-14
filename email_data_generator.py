import json
from libs.client import AiClient

l_sector = [
    'Grocery Stores', 'Restaurants', 'Fast Food Restaurants', 'Pharmacies',
    'Service Stations (Fuel)', 'Electronics Stores', 'Taxi service']
l_city = ['Beijing', 'Paris', 'Bujumbura', 'Berlin', 'Shanghai']
l_size = ['small', 'medium', 'large']

f_prompt = """
Role: You are an expert content writer with extensive direct marketing
experience. You have strong writing skills, creativity, adaptability to
different tones and styles, and a deep understanding of audience needs and
preferences for effective direct campaigns.
Context: You have to write a short message in no more than 2 sentences for a
direct marketing campaign to sell a new ecommerce payment service to stores.
The target stores have the following three characteristics:
- The sector of activity: {sector}
- The city where the stores are located: {city}
- The size of the stores: {size}
Task: Write a short message for the direct marketing campaign. Use the skills
defined in your role to write this message! It is important that the message
you create takes into account the product you are selling and the
characteristics of the store you are writing to.
"""

f_sub_prompt = "{sector}, {city}, {size}"

def main():
    res = []
    ai = AiClient()
    nb_rep = 3

    for sector in l_sector:
        for city in l_city:
            for size in l_size:
                for i in range(nb_rep):  # 'nb_rep' times each example
                    prompt = f_prompt.format(sector=sector, city=city, size=size)
                    sub_prompt = f_sub_prompt.format(sector=sector, city=city, size=size)

                    response_txt = ai.ask([{"role": "user", "content": prompt}])
                    response_txt = response_txt.replace('"', '')
                    print(response_txt)

                    new_row = {'prompt': sub_prompt,'completion': response_txt}
                    new_row = {'messages': [{'role': 'user', 'content': sub_prompt}, {'role': 'assistant', 'content': response_txt}]}
                    res.append(new_row)

    with open('training.jsonl', 'w') as file:
        for entry in res:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')

if __name__ == '__main__':
    main()