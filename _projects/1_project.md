---
layout: page
title: PlanYourMeals.com 2017 - 2020
description: with background image
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---

Every project has a beautiful feature showcase page.
It's easy to include images in a flexible 3-column grid format.
Make your photos 1/3, 2/3, or full width.

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

PlanYourMeals was a passion project that grew out of a frustration with existing calorie trackers like MyFitnessPal and the automatic meal planner EatThisMuch. Like so many people, my early efforts going to gym were limited until that realization that diet is 80% of success. The bottom line? Muscles grow much easier in the precesnse of a caloric surplus and fat only disappears in the precense of a deficit. Working out with controlling eating leads to a lot of effort with marginly noticeable results. 

Originally, I used MyFitnessPal to track calories, but could rarely get a perfect macro fit. Often, I'd find myself out of ideas while trying to find a source of protein for dinner after 'spending' too many carbs and fats at breakfast and lunch. I nexted tried EatThisMuch, a programatic meal planner that combined different recipes to make a macro-optimized daily plan. Although I had paid for the pro version and spent time setting up a menu, the plans I got back were always frustrating - for example, three different dishes for breakfast, each involving prep and cook. Then, a preparead lunch (who has time?) and a vegetable dish as a 'snack'. 

My concept for planyourmeals was a planner that would be able to create plans that adapted to how I might actually during the day, using my frequent meals and snacks most frequently, and that would allow me to make mid-day changes, for example if I no longer had time to cook and wanted to go to Chipotle instead. 

### The Core Model 

I first built an optimization model that would assign integer amounts of realistic serving sizes per food item. The following code sets up a Pyomo modeling object with the following problem: 

Maximize The Sum of relevance Score 


1. 

Disclaimer: this code was written from 2017 - 2018 and I may not make the same choices today. See [full code here](https://github.com/jamesdvance/planyourmealsapi/blob/master/autoplanner/autoplan_week.py#L423)
```python
		has_dj = False
		# Model Create and Index
		model = ConcreteModel()
		# Indices
		access_df = food_df[['access_idx']]
		full_I_dict = access_df['access_idx'].to_dict()
		access_df.index = access_df['access_idx']
		access_df = access_df.assign(full_I=range(len(access_df)))
		access_dict = access_df['full_I'].to_dict()
		model.access_idx = Set(initialize=access_df.index)
		model.full_I = Set(initialize=food_df.index) # used for objective function only
		# Params
		model.access_idx_lkup = Param(model.access_idx, initialize=access_dict) # To lookup the i based on the access_idx
		model.full_I_lkup = Param(model.full_I, initialize=full_I_dict) # To lookup the access_idx based on the i. For parsing solves
		# Sub-indeces of Full_I
		# Going to loop through meal_idx_tup and use setattr
		for tup in meals_idx_tup:
			if incl_dict[tup] and tup[1] != 'Snack':
				# Setting individual index
				setattr(model,tup[0]+"_"+tup[1], Set(initialize=food_df[(food_df['day']==tup[0])&(food_df['meal']==tup[1])].index))
				if src_constr_dict[tup]=='sa':
					setattr(model, tup[0]+"_"+tup[1]+"_wm",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Whole Meals')].index))
				elif src_constr_dict[tup]=='bd':
					setattr(model, tup[0]+"_"+tup[1]+"_mc",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Main Courses')].index))
					setattr(model, tup[0]+"_"+tup[1]+"_si",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Sides')].index))
				else:
					setattr(model, tup[0]+"_"+tup[1]+"_wm",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Whole Meals')].index))
					setattr(model, tup[0]+"_"+tup[1]+"_mc",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Main Courses')].index))
					setattr(model, tup[0]+"_"+tup[1]+"_si",Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                          (food_df['meal']==tup[1])&\
		                                                                         (food_df['dish_num']=='Sides')].index))
			elif incl_dict[tup] and tup[1] == 'Snack':
				setattr(model,tup[0]+"_"+tup[1], Set(initialize=food_df[(food_df['day']==tup[0])&\
		                                                                (food_df['meal']==tup[1])].index))
		"""
		Model Params
		"""
		model.amt_ub = Param(model.full_I, initialize=food_df[['max_servings']].to_dict('dict')['max_servings'])
		"""
		Creating Variables
		"""
		model.ind = Var(model.full_I, within=Binary)
		model.amt = Var(model.full_I, within=NonNegativeIntegers, bounds=(0,food_max))
		# Creating Constraints
		# Set A Global Big M constraint so that ind is tied to amt
		def m(model, i):
			return model.amt[i] <= model.ind[i] * global_m
		def use_ind(model, i):
			return model.ind[i] <= model.amt[i]

		model.m_items = Constraint(model.full_I, rule=m)
		model.use_m = Constraint(model.full_I, rule=use_ind)

		"""
		Don't repeat accross days
		"""
		#for tup in meal_food_tup.itertuples():
		for tup in meal_food_tup:
			"""
			Does not include meals
			"""
			# print(tup)
			# idx = food_df[(food_df['fd_type']=='food')&(food_df['unique_id']==tup[1])].index
			# setattr(model, str(tup[1])+"_non_rep", Constraint(expr=sum([model.ind[m] for m in idx])<=tup[2])) # Sets all to '2' which is highly infeasible

			idx = food_df[(food_df['meal']==tup[0])&(food_df['fd_type']==tup[1])&(food_df['unique_id']==tup[2])].index
			setattr(model, tup[0]+tup[1]+str(tup[2])+"_non_rep", Constraint(expr=sum([model.ind[m] for m in idx])<=2)) 
		
		"""
		Disjunction
		"""
		for tup in meals_idx_tup: # Can combine with above loop after getting this down
			if incl_dict[tup] and tup[1] != 'Snack':        
				if src_constr_dict[tup]=='sa':
					setattr(model, tup[0]+"_"+tup[1]+"_sel_wm", 
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_wm")])==1))
				elif src_constr_dict[tup]=='bd':
					setattr(model,tup[0]+"_"+tup[1]+"_sel_mc", 
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_mc")])==1))
					setattr(model,tup[0]+"_"+tup[1]+"_sel_si", 
						Constraint(expr=inequality(sides_dict['br']['min'],sum([model.ind[m] for m in getattr(model,
		                                                                                    tup[0]+"_"+tup[1]+"_si")]))))
				else:
					setattr(model,tup[0]+"_"+tup[1]+"_sel_sa", 
						Disjunct())
					setattr(getattr(model,tup[0]+"_"+tup[1]+"_sel_sa"),'c1', 
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_wm")])==1))
					setattr(getattr(model,tup[0]+"_"+tup[1]+"_sel_sa") ,'c2',
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_mc") ])==0))
					setattr(getattr(model,tup[0]+"_"+tup[1]+"_sel_sa"),'c3',
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_si") ])==0))
					setattr(model,tup[0]+"_"+tup[1]+"_sel_bd", 
						Disjunct())
					setattr(getattr(model,tup[0]+"_"+tup[1]+"_sel_bd"),'c1',
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_wm") ])==0))
					setattr(getattr(model, tup[0]+"_"+tup[1]+"_sel_bd"),'c2',
						Constraint(expr=sum([model.ind[m] for m in getattr(model,tup[0]+"_"+tup[1]+"_mc") ])==1))
					setattr(getattr(model,tup[0]+"_"+tup[1]+"_sel_bd"),'c3',
						Constraint(expr=inequality(sides_dict[tup[1]]['min'],sum([model.ind[m] for m in getattr(model, 
						tup[0]+"_"+tup[1]+"_si" )]),sides_dict[tup[1]]['max'])))
					setattr(model,tup[0]+"_"+tup[1]+"c",Disjunction(expr=[getattr(model,tup[0]+"_"+tup[1]+"_sel_sa" ), 
						getattr(model,tup[0]+"_"+tup[1]+"_sel_bd")]))
					has_dj = True
			elif incl_dict[tup] and tup[1] == 'Snack':
				# Snack Amount Constraints
				setattr(model, tup[0]+"_"+tup[1]+"_ind_n",Constraint(expr=sum([model.ind[m] for m in getattr(model,
					tup[0]+"_"+tup[1])])<=n_snack) )
		# Disjunctive Transformation
		if has_dj:
			trnsfrm = TransformationFactory('gdp.bigm') # Toggle chull bigm
			trnsfrm.apply_to(model)

		# Amount max
		def amt_bounds(model,i):
			return model.amt[i] <= model.amt_ub[i]
		model.amt_bounds = Constraint(model.full_I, rule=amt_bounds)
		# Nutritional bounds  
		for day in days:
			if day in leftover_dict:
				setattr(model, day+"_index", Set(initialize=food_df[(food_df['day']==day)|\
					((food_df['day']==leftover_dict[day]['orig_day'])&\
					(food_df['meal']==leftover_dict[day]['orig_meal']))].index))
			else:
				setattr(model, day+"_index", Set(initialize=food_df[(food_df['day']==day)].index))
			# Will need to also change food_df_dict going forward for different days that already have something planned
			# Set Index N
			nutrient_df = pd.DataFrame(requirements_dict[day])
			nut_cols = list(nutrient_df.columns.values)
			setattr(model, day+"_N", Set(initialize=nut_cols)) 
			food_df_dict = food_df[['prob_r']+nut_cols].to_dict('list')
			setattr(model, day+"_bounds", Param(getattr(model,day+"_N"), initialize=nutrient_df[nut_cols].to_dict('list'))) 
			
			def ml_bounds(model, n):
				return (getattr(model, day+"_bounds")[n][0], sum([food_df_dict[n][i]* model.amt[i] for i in getattr(model,day+"_index")]), 
					getattr(model, day+"_bounds")[n][1])
			setattr(model, day+"_nutr_bounds", Constraint(getattr(model,day+"_N"), rule=ml_bounds))

		# Objective
		def tgt_obj(model):
			return  sum_product(food_df_dict['prob_r'], model.amt, index=model.full_I)
		model.obj = Objective(rule=tgt_obj)
```

## Learnings

### Skills
1. Pyomo, CBC, Branch-and-Cut Optimization methods
2. Django, Postgresql
3. React, Redux and State
4. Celery and asyncronous workflows
5. Nginx, gunicorn, and Linux hosting (etcd, port management, etc)
6. AWS Cloud (EC2, S3, CodeBuild, CodePipeline, VPC's, Subnets, RDS, etc)

### Project-Level
1. Appreciation for Frontend

The experience of understanding React modules, the choices for managing state, and how bottlenecked the entire flow was by the frontend was an eye-opening experience and a 

2. Data model choices anchor your project, are hard to change

3. Data Categories, Tags and Labels are Crucial

4. Data Science Proejcts are Continuous

Within months of building my POC, I had a backlog of data science-based tasks. Semantic similarity between serving size descriptions. Named entity recognition to parse free text descriptions into ingredient unit, amount and serving size. K-nearest neighbors for similar-nutrient profile recommendations. Image classification or similar search to identify incorrect item thumbnails. 

5. Good Software Is Not a One-Person Job
I wanted to upskill my data science by finding 

## Conclusion
This project evolved from a desire to dig deeper into Optimization into a part-time obsession with making the best meal planning project. In the end, there wasn't much of a clamouring for a hyper-precise adpable meal planner. With the arrival of my first daughter, and a new job during the Covid-19 pandemic I decided to simplify my life and step back from the project, investing the additional energy into my full-time job. 

I'm happy with that decision, but still proud of the work and the learnings along the way. 

# Appendix

## FAQ's

**Why is the code all added to github at once?**

While the code originally was pushed to github, in 2017 I switched to AWS CodeCommit for greater security and CICD integration with AWS native tools. In 2025, 