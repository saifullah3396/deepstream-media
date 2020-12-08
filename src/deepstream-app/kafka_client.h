#ifndef __KAFKA_CLIENT_H__
#define __KAFKA_CLIENT_H__

#include <csignal>
#include <cppkafka/cppkafka.h>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <stdexcept>

using namespace std;
using namespace cppkafka;

#ifdef __cplusplus
extern "C"
{
#endif

class KafkaClient {
public:
  KafkaClient() {
  }
  ~KafkaClient() {
    delete consumer;
  }

  void configure() {
    std::cout << "Configuring kafka client..." << std::endl;
    // Construct the configuration
    config = {
        { "metadata.broker.list", brokers },
        // Enable auto commit
        { "enable.auto.commit", true },
        // Add group
        { "group.id", "message-receiver"},
    };
  }

  void init_consumer() {
    std::cout << "Initializing kafka consumer..." << std::endl;

    // Create the consumer
    consumer = new Consumer(config);

    // Print the assigned partitions on assignment
    consumer->set_assignment_callback([](const TopicPartitionList& partitions) {
        cout << "Got assigned: " << partitions << endl;
    });

    // Print the revoked partitions on revocation
    consumer->set_revocation_callback([](const TopicPartitionList& partitions) {
        cout << "Got revoked: " << partitions << endl;
    });

    std::cout << "Subscribing to topic..." << std::endl;
    // Subscribe to the topic
    consumer->subscribe({ topic_name });
  }

  Json::Value consume() {
    try {
      // Try to consume a message
      Message msg = consumer->poll();
      if (msg) {
        // If we managed to get a message
        if (msg.get_error()) {
          // Ignore EOF notifications from rdkafka
          if (!msg.is_eof()) {
              cout << "[+] Received error notification: " << msg.get_error() << endl;
          }
        }
        else {
          std::stringstream sstr(msg.get_payload());
          Json::Value json;
          sstr >> json;
          return json;
        }
      }
      return Json::nullValue;
    } catch (Json::Exception& e) {
      std::cerr << 
        "Invalid msg received!" << std::endl;
      return Json::nullValue;
    }
  }

private:
  string brokers {"10.12.42.157:9092"};
  string topic_name {"camera"};

  // Construct the configuration
  Configuration config;
  Consumer* consumer;
};

#ifdef __cplusplus
}
#endif

#endif // __KAFKA_CLIENT_H__
