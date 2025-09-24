
- Always write code and comments in Python and English
- Follows KISS principle: Keep it simple, stupid
- Follows YAGNI principle: You Aren't Gonna Need It 
- Every time, after the initial try, always follow by a step with eliminating unnecessary complexity
- Minimize the code revisions. for example:
  * if I ask you to re-orgnize different code parts, you should try to avoid modify the code
  * if I ask you to update or change part of the functions, try the best to avoid revising other non relevant parts unless it is necessary
- Prioritize performance-optimized solutions.
- Keep responses concise unless asked for more details.
- Keep responses as working in windows enviroment, like powershell (not linux)


- all comments in the code should be in English

- i prefer fewer lines of codes, as soon as they are readable and understandable. for example, I prefer  
{'id': 'walkforward', 'name': 'WalkForward', 'type': 'walk_forward', 'train_days': 252, 'test_days': 63}
don't prefer:
    {
        'id': 'walkforward',
        'name': 'WalkForward',
        'type': 'walk_forward',
        'train_days': 252,
        'test_days': 63
    }

- don't add unnecessary print in the codes, and try to combine them in fewer line of codes for print results or prompts

- keep MVP concepts and occam razor principle - don't add codes or new features unless they are necessary or will significantly improve the project

- use OOP and functional programming when they could significantly improve the efficiency

- for logging, never write the verify or check or test steps - only keep the coding related work

- to avoid path issue, DON'T use python .\src\features.py, use python -m src.features