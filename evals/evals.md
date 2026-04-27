# Evaluations

## 1. Automating prompt engineering

|     | Agent                                                                                                                             | Metric    | Judge LM | Teacher LM | Task LM    | Score (devset) | Score (testset) |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------- | :-------- | :------- | :--------- | :--------- | :------------- | :-------------- |
| 1.1 | [ecb_hr_expert_agent, unoptimized, gpt-5](./1_automating_prompt_engineering/1_1_ecb_hr_expert_agent_unoptimized_gpt_5/)             | composite | gpt-5    | n/a        | gpt-5      | 51.55          | 47.64           |
| 1.2 | [ecb_hr_expert_agent, optimized, gpt-5](./1_automating_prompt_engineering/1_2_ecb_hr_expert_agent_optimized_gpt_5/)                 | composite | gpt-5    | gpt-5      | gpt-5      | 88.00          | 79.32           |

## 2. Overcoming high sensitivity to LMs

|     | Agent                                                                                                                             | Metric    | Judge LM | Teacher LM | Task LM    | Score (devset) | Score (testset) |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------- | :-------- | :------- | :--------- | :--------- | :------------- | :-------------- |
| 2.1 | [ecb_hr_expert_agent, unoptimized, gpt-4o](./2_overcoming_high_sensitivity_to_lms/2_1_ecb_hr_expert_agent_unoptimized_gpt_4o/)      | composite | gpt-5    | n/a        | gpt-4o     | 46.00          | 43.20           |
| 2.2 | [ecb_hr_expert_agent, optimized, gpt-4o](./2_overcoming_high_sensitivity_to_lms/2_2_ecb_hr_expert_agent_optimized_gpt_4o/)          | composite | gpt-5    | gpt-4o     | gpt-4o     | 71.30          | 62.00           |
| 2.3 | ecb_hr_expert_agent, unoptimized, gpt-5 (same as 1.1)                                                                             | composite | gpt-5    | n/a        | gpt-5      | 51.55          | 47.64           |
| 2.4 | ecb_hr_expert_agent, optimized, gpt-5 (same as 1.2)                                                                               | composite | gpt-5    | gpt-5      | gpt-5      | 88.00          | 79.32           |

## 3. Enabling the use of smaller LMs

|     | Agent                                                                                                                             | Metric    | Judge LM | Teacher LM | Task LM    | Score (devset) | Score (testset) |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------- | :-------- | :------- | :--------- | :--------- | :------------- | :-------------- |
| 3.1 | ecb_hr_expert_agent, unoptimized, gpt-5 (same as 1.1)                                                                             | composite | gpt-5    | n/a        | gpt-5      | 51.55          | 47.64           |
| 3.2 | ecb_hr_expert_agent, optimized, gpt-5 (same as 1.2)                                                                               | composite | gpt-5    | gpt-5      | gpt-5      | 88.00          | 79.32           |
| 3.3 | [ecb_hr_expert_agent, unoptimized, gpt-5-mini](./3_enabling_the_use_of_smaller_lms/3_3_ecb_hr_expert_agent_unoptimized_gpt_5_mini/) | composite | gpt-5    | n/a        | gpt-5-mini | 45.95          | 45.56           |
| 3.4 | [ecb_hr_expert_agent, optimized, gpt-5-mini](./3_enabling_the_use_of_smaller_lms/3_4_ecb_hr_expert_agent_optimized_gpt_5_mini/)     | composite | gpt-5    | gpt-5      | gpt-5-mini | 88.20          | 76.60           |
| 3.5 | [ecb_hr_expert_agent, unoptimized, gpt-5-nano](./3_enabling_the_use_of_smaller_lms/3_5_ecb_hr_expert_agent_unoptimized_gpt_5_nano/) | composite | gpt-5    | n/a        | gpt-5-nano | 43.60          | 39.88           |
| 3.6 | [ecb_hr_expert_agent, optimized, gpt-5-nano](./3_enabling_the_use_of_smaller_lms/3_6_ecb_hr_expert_agent_optimized_gpt_5_nano/)     | composite | gpt-5    | gpt-5      | gpt-5-nano | 80.45          | 62.40           |
