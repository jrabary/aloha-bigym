## Filemap

[aloha_box.py](aloha_box.py) contains `StoreBox` and `PickBox`
<div>
  <img width="20%" src="images/491130E7-CEBC-4499-A6F8-0EBF80A7EC2A_1_201_a.jpeg" >
</div>

[aloha_cupboards.py](aloha_cupboards.py) contains `WallCupboardOpen`, `WallCupboardClose`, `DrawerTopOpen`, `DrawerTopClose`, `DrawersAllOpen`, `DrawersAllClose`
<div>
  <img width="20%" src="images/14B6FAEB-D88C-4271-8579-1DCC325F317C_1_201_a.jpeg" >
</div>

[aloha_dishwasher.py](aloha_dishwasher.py) contains `DishwasherOpen`, `DishwasherClose`, `DishwasherOpenTrays`, `DishwasherCloseTrays`

[aloha_dishwasher_cups.py](aloha_dishwasher_cups.py) contains `DishwasherUnloadCups`, `DishwasherLoadCups`, `DishwasherUnloadCupsLong`
<div>
  <img width="20%" src="images/3C23169B-D71D-4701-9FC6-3F5742DB1026_1_201_a.jpeg" >
</div>

[aloha_dishwasherplates.py](aloha_dishwasherplates.py) contains `DishwasherUnloadPlates`, `DishwasherLoadPlates`, `DishwasherUnloadPlatesLong`
<div>
  <img width="20%" src="images/B3590652-E684-4B08-81FE-99558B864777_1_201_a.jpeg" >
</div>

[aloha_dishwashercutlery.py](aloha_dishwashercutlery.py) contains `DishwasherUnloadCutlery`, `DishwasherLoadCutlery`, `DishwasherUnloadCutleryLong`

[aloha_dualreachtar.py](aloha_dualreachtar.py) contains `ReachTargetDual`, `ReachTarget`, `ReachTargetSingle`
<div>
  <img width="20%" src="images/12E4435B-2096-4F53-A950-94B48E462618_1_201_a.jpeg" >
</div>

[aloha_flip.py](aloha_flip.py) contains `FlipCup`, `FlipCutlery`

[aloha_groceries.py](aloha_groceries.py) contains `GroceriesStore`
<div>
  <img width="20%" src="images/Screenshot 2024-09-01 at 10.53.42 PM.png" >
</div>

[aloha_moveplates.py](aloha_moveplates.py) contains `MovePlate`, `MoveTwoPlates`

[aloha_saucepan.py](aloha_saucepan.py) contains `FlipSandwich`, `ToastSandwich`, `SaucepanToHob`, `RemoveSandwich`
<div>
  <img width="20%" src="images/2F26601D-258E-4A11-8293-E68A19A8B732_1_201_a.jpeg" >
</div>

[aloha_stackblocks.py](aloha_stackblocks.py) contains `StackBlocks` 
<div>
  <img width="20%" src="images/99002BF3-BFFC-43CA-99C6-0D6AE7983F0F_1_201_a.jpeg" >
</div>

[aloha_storekitchenware.py](aloha_storekitchenware.py) contains `StoreKitchenware`

[aloha_takecups.py](aloha_takecups.py) contains `TakeCups`, `PutCups`

<div>
  <img width="20%" src="images/Screenshot 2024-09-01 at 10.56.47 PM.png" >
</div>

No longer relevant: `PickBox`, `GroceriesStoreLower`, `CupboardsOpenAll`, `CupboardsCloseAll`


# Common Errors

If this error is encountered: 

```
from mojo.elements import Body, Site, MujocoElement
ImportError: cannot import name 'MujocoElement' from 'mojo.elements'
```

Add this line to `site-packages/mojo/elements/__init__.py` in your python directory:

`from mojo.elements.element import MujocoElement`