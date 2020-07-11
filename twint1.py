import nest_asyncio
nest_asyncio.apply()

import twint

c = twint.Config()
c.Search = "coronavirus"

c.Since="2020-07-05 00:00:00"
c.Until="2020-07-06 00:00:00"
c.Verified=False
c.Store_csv = True
c.Output = "tweet2"
twint.run.Search(c)




