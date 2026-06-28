"""Publish (or update) the custom feed so it appears in the Bluesky app.

Writes an app.bsky.feed.generator record into YOUR repo, pointing at the feed
service DID (did:web:<tunnel-host>). Re-run this whenever the tunnel hostname
(and thus the service DID) changes — same rkey means it updates in place.

Usage:
    python publish_feed.py            # publish/update
    python publish_feed.py --unpublish
"""

from __future__ import annotations

import argparse

from atproto import models

import config
from auth import get_client


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the DiscourseNuke feed.")
    ap.add_argument("--unpublish", action="store_true", help="Remove the feed record.")
    args = ap.parse_args()

    client = get_client()
    print(f"Authenticated as {client.me.handle} ({client.me.did})")

    if args.unpublish:
        client.com.atproto.repo.delete_record(
            models.ComAtprotoRepoDeleteRecord.Data(
                repo=client.me.did,
                collection=models.ids.AppBskyFeedGenerator,
                rkey=config.FEED_RKEY,
            )
        )
        print(f"Unpublished feed '{config.FEED_RKEY}'.")
        return

    if config.FEEDGEN_HOSTNAME == "example.invalid":
        raise SystemExit(
            "FEEDGEN_HOSTNAME is not set. Put your tunnel host in .env first."
        )

    record = models.AppBskyFeedGenerator.Record(
        did=config.FEEDGEN_SERVICE_DID,
        display_name=config.FEED_DISPLAY_NAME,
        description=config.FEED_DESCRIPTION,
        created_at=client.get_current_time_iso(),
    )
    resp = client.com.atproto.repo.put_record(
        models.ComAtprotoRepoPutRecord.Data(
            repo=client.me.did,
            collection=models.ids.AppBskyFeedGenerator,
            rkey=config.FEED_RKEY,
            record=record,
        )
    )
    print(f"Published feed -> {resp.uri}")
    print(f"  service DID : {config.FEEDGEN_SERVICE_DID}")
    print(f"  endpoint    : {config.SERVICE_ENDPOINT}")
    print()
    print("Next steps:")
    print(f"  1. Add this to .env:  FEEDGEN_PUBLISHER_DID={client.me.did}")
    print("  2. Restart the server so describeFeedGenerator reports the feed URI.")
    print("  3. Open the feed in the Bluesky app and pin it:")
    print(f"     https://bsky.app/profile/{client.me.did}/feed/{config.FEED_RKEY}")


if __name__ == "__main__":
    main()
