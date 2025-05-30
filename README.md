## Explanation

This is the respository for the paper *RZ-NAS: Enhancing LLM-guided Neural Architecture Search via Reflective Zero-Cost Strategy* . One example of the whole prompt is saved in `template.txt`. More zero-cost proxies and search spaces are saved under the folder `descriptions`. 


## Search

For different zero-cost proxies, you can change the parameter `zero_shot_score`.

```
python evolution_search.py --gpu 0 --zero_shot_score <zero-cost proxy> --search_space <micro/macro search space> 
```

more customized parameters setting can be found in ./scripts.



