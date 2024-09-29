# Surpassing Named Entity Recognition With LLMs

[Named Entity Recognization](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER) is a technique to identify and classify entities within textual data. For example While working on 'PlanYourMeals', I wanted to parse free text showing ingredients for 'units'. A named entity recognition model helped to (imperfectly) parse 'a cup of sugar' into {"item": "sugar", "unit": cup, "quantity: 1}. One of the hardest parts about this task was how words can multiple meanings even in similar contexts. For example, '1 fruit cup' in a Pim's cup recipe should be parsed to {"item": "fruit cup", "unit": "item", "quantity": 1} and a Grande Latte's '16 oz cup' should map to {"item": latte", "unit": "oz", "quantity": 16} 

This problem manifests again in the ['Learning Agency Lab' PII Data Competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data). 
