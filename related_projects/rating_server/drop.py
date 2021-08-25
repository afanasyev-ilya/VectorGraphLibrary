from web_rating.lib import mongo_api
import sys

if sys.argv[1] == "print":
    mongo_api.print_collection()
elif sys.argv[1] == "remove_all":
    mongo_api.drop_collection()
else:
    mongo_api.remove_one(sys.argv[1])
